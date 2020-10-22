"""
Prepare train and validation splits
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


# TODO: Generate train and validate splits using stratification.
# TODO: Store train, validate splits in train and valid folders
# TODO: Label as malicious or benign (binary)
# TODO: Store and version datasets using DVC
# TODO: Repeat this for a mini-dga dataset
# TODO: Create test dataset


def prepare_train_test(source_path, dataset_file, test_size=0.10):
    dataset_df = pd.read_csv(source_path / dataset_file)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_df['domain'],
        dataset_df['label'],
        test_size=test_size,
        stratify=dataset_df['label']
    )
    train_valid_df = X_train.to_frame().join(y_train)
    test_df = X_test.to_frame().join(y_test)
    return train_valid_df, test_df


def prepare_train_valid(train_valid_df, valid_size=0.10):
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_valid_df['domain'],
        train_valid_df['label'],
        test_size=valid_size,
        stratify=train_valid_df['label']
    )
    train_df = X_train.to_frame().join(y_train)
    valid_df = X_valid.to_frame().join(y_valid)
    return train_df, valid_df


def encode_labels(df, mappings):
    return df.replace(mappings)


def main(repo_path, prefix, dataset_file, test_size, valid_size):
    # setup paths
    data_path = repo_path / "data"
    processed_path = data_path / "processed"
    # stratified train valid split
    train_valid_df, test_df = prepare_train_test(processed_path, dataset_file, test_size=test_size)
    train_df, valid_df = prepare_train_valid(train_valid_df, valid_size=0.10)
    # label encode
    l_encode = {
        'banjori': 1, 'gozi': 1, 'locky': 1,
        'matsnu': 1, 'murofet': 1, 'necurs': 1,
        'conficker': 1, 'beebone': 1, 'chinad': 1,
        'volatilecedar': 1, 'alexa_top_1m': 0, 'majestic_million': 0
    }
    enc_train_df = encode_labels(train_df, l_encode)
    enc_valid_df = encode_labels(valid_df, l_encode)
    enc_test_df = encode_labels(test_df, l_encode)

    # save train and valid datasets
    enc_train_df.to_csv(processed_path / f"{prefix}train.csv", index=False)
    enc_valid_df.to_csv(processed_path / f"{prefix}validation.csv", index=False)
    enc_test_df.to_csv(processed_path / f"{prefix}test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='dataset.csv')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--test_size', type=float, default=0.1, metavar='N',
                        help='Percentage size of test dataset (default: 10%)')
    parser.add_argument('--valid_size', type=float, default=0.2, metavar='N',
                        help='Percentage size of validation dataset (default: 20%)')
    args = parser.parse_args()
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path, args.prefix, args.ds_name, args.test_size, args.valid_size)
