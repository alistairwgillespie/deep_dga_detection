import optuna
import os
import joblib
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score
from dga.datasets import DomainDataset
from torch.utils.data import DataLoader
from dga.models.dga_classifier import DGAClassifier
import mlflow
import mlflow.pytorch


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, yy, x_lens, y_lens


def prepare_dl(data_dir, filename, batch_size):
    df = pd.read_csv(os.path.join(data_dir, filename))
    ds = DomainDataset(df, train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    return dl


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def calc_prec_rec(y_pred, y_actual):
    prec = precision_score(y_actual, y_pred)
    rec = recall_score(y_actual, y_pred)
    return prec, rec


def train(log_interval, model, epoch, train_loader, optimizer, criterion):
    model.train()
    train_pred = np.array([])
    train_actual = np.array([])
    total_loss = 0.
    for batch_num, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_padded, x_lens)
        loss = criterion(output, torch.Tensor(y_padded).unsqueeze(1))
        train_pred = np.append(train_pred, output.detach().cpu().numpy())
        train_pred = np.round(train_pred).astype(int)
        train_actual = np.append(train_actual, y_padded).astype(int)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_num % log_interval == 0:
            print(f'Train Epoch: {epoch}' 
                    f'[{batch_num*len(x_padded)}/{len(train_loader.dataset)}'
                    f' ({100. * batch_num / len(train_loader):.0f}%)]'
                    f'\tLoss: {loss.item():.6f}')

    accuracy = (train_pred == train_actual).sum()/len(train_loader.dataset)
    prec, rec = calc_prec_rec(train_pred, train_actual)
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(train_actual, train_pred, average='micro')
    print(
        f"Train      | "
        f"Epoch {epoch:03} | "
        # f"Batch {batch_num:03} | "
        f"Loss {avg_loss:.3f} | "
        f"Accuracy {accuracy:.3f} | "
        f"Precision {prec:.3f} | "
        f"Recall {rec:.3f} | "
        f"F1 {f1:.3f} "
    )

    mlflow.log_metric(key="train_accuracy", value=accuracy, step=epoch)
    mlflow.log_metric(key="train_precision", value=prec, step=epoch)
    mlflow.log_metric(key="train_recall", value=rec, step=epoch)
    mlflow.log_metric(key="train_loss", value=avg_loss, step=epoch)


def validate(model, epoch, valid_loader, criterion):
    model.eval()
    val_pred = np.array([])
    val_actual = np.array([])
    valid_loss = 0.
    with torch.no_grad():
        for batch_num, (x_padded, y_padded, x_lens, y_lens) in enumerate(valid_loader):
            output = model(x_padded, x_lens)
            val_pred = np.append(val_pred, output.detach().cpu().numpy())
            val_pred = np.round(val_pred).astype(int)
            val_actual = np.append(val_actual, y_padded).astype(int)
            loss = criterion(output, torch.Tensor(y_padded).unsqueeze(1))
            valid_loss += loss.item()

    accuracy = (val_pred == val_actual).sum() / len(valid_loader.dataset)
    prec, rec = calc_prec_rec(val_pred, val_actual)
    f1 = f1_score(val_actual, val_pred, average='micro')
    avg_loss = valid_loss / len(valid_loader)

    print(
        f"Validation | "
        f"Epoch {epoch:03} | "
        f"Loss {avg_loss:.3f} | "
        f"Accuracy {accuracy:.3f} | "
        f"Precision {prec:.3f} | "
        f"Recall {rec:.3f} | "
        f"F1 {f1:.3f} "
    )

    mlflow.log_metric(key="valid_accuracy", value=accuracy, step=epoch)
    mlflow.log_metric(key="valid_precision", value=prec, step=epoch)
    mlflow.log_metric(key="valid_recall", value=rec, step=epoch)
    mlflow.log_metric(key="valid_loss", value=valid_loss, step=epoch)
    mlflow.log_metric(key="valid_f1", value=f1, step=epoch)

    return f1


def hyper_train(trial):

    config = {
        'output_data_dir': 'data/',
        'model_dir': 'models/',
        'data_dir': 'data/processed/',
        'train_fn': 'mini_train.csv',
        'validation_fn': 'mini_validation.csv',
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'train_batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 1,
        'seed': 0,
        'input_features': 68,
        'log_interval': 100,
        'save_model': False,
        'lr': trial.suggest_loguniform('lr', 1e-3, 1e-2),
        'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
        'hidden_dim': trial.suggest_int('hidden_dim', 10, 50),
        'embedding_dim': trial.suggest_int('embedding_dim', 3, 30),
        'n_layers': trial.suggest_int('n_layers', 2, 10),
        'optimizer': trial.suggest_categorical('optimizer', [optim.SGD, optim.RMSprop, optim.Adam]),
        'save_model': False
    }

    # Args holds all passed-in arguments
    device = torch.device(config['device'])
    cuda = torch.cuda.is_available()
    print("Using device {}.".format(device))
    torch.manual_seed(config['seed'])

    # Get loaders
    train_loader = prepare_dl(config['data_dir'], config['train_fn'], config['train_batch_size'])
    valid_loader = prepare_dl(config['data_dir'], config['validation_fn'], config['test_batch_size'])

    # Initialize DGA Classifier
    model = DGAClassifier(input_features=config['input_features'],
                          hidden_dim=config['hidden_dim'],
                          n_layers=config['n_layers'],
                          embedding_dim=config['embedding_dim'],
                          batch_size=config['train_batch_size'],
                          dropout_rate=config['dropout_rate'])

    if cuda:
        model.cuda()

    # Define loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

    train_start = time.time()  # start timer

    # Commence training
    with mlflow.start_run() as run:
        for key, value in config.items():
            mlflow.log_param(key, value)
        for epoch in range(1, config['epochs'] + 1):
            train(config['log_interval'], model, epoch, train_loader, optimizer, criterion)
            f1 = validate(model, epoch, valid_loader, criterion)

        # mlflow.pytorch.log_model(model, "models")

    train_finish = time.time()  # end timer
    duration_secs = train_finish - train_start
    duration_time = str(datetime.timedelta(seconds=duration_secs))
    print(f'[DURATION]: {duration_time}')

    # Save the model parameters
    if config["save_model"]:
        model_path = os.path.join(config['model_dir'], 'dga_model.pth')
        with open(model_path, 'wb') as f:
            torch.save(model.cpu().state_dict(), f)

    return f1


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=hyper_train, n_trials=150)
    joblib.dump(study, 'studies/dga_optuna.pkl')

# %%
import optuna
import joblib
study = joblib.load('../studies/dga_optuna.pkl')
df = study.trials_dataframe()
df.sort_values(by=['value'], ascending=False)

# %%
