import os
import argparse
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score
from dga.datasets import DomainDataset
from torch.utils.data import DataLoader
from dga.models.dga_classifier import DGAClassifier
import mlflow
import mlflow.pytorch

# TODO: Implement cross validation i.e. K-Fold Cross-Validation
# TODO: Early stop to prevent overfitting
# TODO: Ensembling with sub-domain model and feature set
# TODO: Baseline model (Entropy)


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


def train(epoch):
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
        if batch_num % args.log_interval == 0:
            print(f'Train Epoch: {epoch}' 
                    f'[{batch_num*len(x_padded)}/{len(train_loader.dataset)}'
                    f' ({100. * batch_num / len(train_loader):.0f}%)]'
                    f'\tLoss: {loss.item():.6f}')

    accuracy = (train_pred == train_actual).sum()/len(train_loader.dataset)
    prec, rec = calc_prec_rec(train_pred, train_actual)
    avg_loss = total_loss / len(train_loader)
    print(
        f"Train Summary     | "
        f"Epoch {epoch:03} | "
        # f"Batch {batch_num:03} | "
        f"Loss {avg_loss:.3f} | "
        f"Accuracy {accuracy:.3f} | "
        f"Precision {prec:.3f} | "
        f"Recall {rec:.3f} "
    )

    mlflow.log_metric(key="train_accuracy", value=accuracy, step=epoch)
    mlflow.log_metric(key="train_precision", value=prec, step=epoch)
    mlflow.log_metric(key="train_recall", value=rec, step=epoch)
    mlflow.log_metric(key="train_loss", value=avg_loss, step=epoch)


def validate(epoch):
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
            print(f"[Batch]: {batch_num}/{len(valid_loader)}", end='\r', flush=True)

    accuracy = (val_pred == val_actual).sum() / len(valid_loader.dataset)
    prec, rec = calc_prec_rec(val_pred, val_actual)
    avg_loss = valid_loss / len(valid_loader)

    print(
        f"Validation Summary | "
        f"Epoch {epoch:03} | "
        f"Loss {avg_loss:.3f} | "
        f"Accuracy {accuracy:.3f} | "
        f"Precision {prec:.3f} | "
        f"Recall {rec:.3f} "
    )

    mlflow.log_metric(key="valid_accuracy", value=accuracy, step=epoch)
    mlflow.log_metric(key="valid_precision", value=prec, step=epoch)
    mlflow.log_metric(key="valid_recall", value=rec, step=epoch)
    mlflow.log_metric(key="valid_loss", value=valid_loss, step=epoch)


if __name__ == '__main__':
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # Default Parameters
    parser.add_argument('--output_data_dir', type=str, default='data/')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--data_dir', type=str, default='data/processed/')
    parser.add_argument('--train_fn', type=str, default='train.csv')
    parser.add_argument('--validation_fn', type=str, default='validation.csv')

    # Training Parameters, given
    parser.add_argument('--log_interval', type=int, default=500, metavar='N',
                        help='log interval (default: 500)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default:5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--input_features', type=int, default=68, metavar='N',
                        help='size of the feature space (default: 67)')
    parser.add_argument('--hidden_dim', type=int, default=30, metavar='N',
                        help='size of the hidden dimension (default: 30)')
    parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                        help='number of hidden layers (default: 2)')
    parser.add_argument('--embedding_dim', type=int, default=10, metavar='N',
                        help='size of the embedding dimension (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='N',
                        help='size of the output dimension (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='dropout rate (default: 0.3)')

    # Args holds all passed-in arguments
    args = parser.parse_args()
    device = torch.device("cpu")
    cuda = torch.cuda.is_available()
    print("Using device {}.".format(device))
    torch.manual_seed(args.seed)

    train_loader = prepare_dl(args.data_dir, args.train_fn, args.batch_size)
    valid_loader = prepare_dl(args.data_dir, args.validation_fn, args.batch_size)

    # Initialize DGA Classifier
    model = DGAClassifier(input_features=args.input_features,
                          hidden_dim=args.hidden_dim,
                          n_layers=args.n_layers,
                          output_dim=args.output_dim,
                          embedding_dim=args.embedding_dim,
                          batch_size=args.batch_size,
                          dropout_rate=args.dropout_rate)

    if cuda:
        model.cuda()

    # Define loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_start = time.time()  # start timer

    # Commence training
    with mlflow.start_run() as run:
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        # train(args.epochs)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            validate(epoch)

        # mlflow.pytorch.log_model(model, "models")

    train_finish = time.time()  # end timer
    duration_secs = train_finish - train_start
    duration_time = str(datetime.timedelta(seconds=duration_secs))
    print(f'[DURATION]: {duration_time}')

    # Save model
    now = datetime.datetime.now()
    ts = str(now.strftime("%Y%m%d_%H-%M-%S"))
    model_info_path = os.path.join(args.model_dir, f'{ts}_dga_model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers,
            'embedding_dim': args.embedding_dim,
            'batch_size': args.batch_size,
            'output_dim': args.output_dim,
            'dropout_rate': args.dropout_rate
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, f'{ts}_dga_model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
