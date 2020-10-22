"""
Make dataset pipeline
"""
from pathlib import Path
import pandas as pd
import csv
import argparse
import subprocess


# TODO: Handle different datasets by inferring domain filed
def get_benign_data(source_path, benign_ds, n=500000):
    domains = []
    labels = []
    for benign in benign_ds:
        max_records = 0
        with open(source_path / benign, 'r') as textfile:
            print(f'Processing: {benign}')
            if "majestic" in benign:
                ix = 2
            else:
                ix = 1
            for row in csv.reader(textfile):
                domains.append(row[ix])
                labels.append(benign.split('.csv')[0])
                max_records += 1
                if max_records == n:
                    break
    data_dictionary = {"domain": domains, "label": labels}
    benign_df = pd.DataFrame(data_dictionary)
    benign_df.drop_duplicates(subset=['domain'], inplace=True)
    return benign_df


def get_dga_data(source_path, dga_ds, n=100000):
    domains = []
    labels = []
    for dga in dga_ds:
        print(f'Processing: {dga}')
        result = subprocess.run(
            ["tail", f"-n {n}", f"{source_path / dga}"],
            stdout=subprocess.PIPE,
            encoding='utf-8')
        for row in csv.reader(result.stdout.splitlines(), delimiter=","):
            domains.append(row[0])
            labels.append(dga.split('_dga.')[0])

    data_dictionary = {"domain": domains, "label": labels}
    dga_df = pd.DataFrame(data_dictionary)
    dga_df.drop_duplicates(subset=['domain'], inplace=True)
    return dga_df


def save_as_csv(data_frame, destination):
    data_frame.to_csv(destination, index=False)


def main(repo_path, ds_name, n_ben, n_dga, shuffle=False):
    # specify dga families and benign data to gather from
    dga_ds = [
        'banjori_dga.csv', 'gozi_dga.csv', 'locky_dga.csv',
        'matsnu_dga.csv', 'murofet_dga.csv', 'necurs_dga.csv',
        'conficker_dga.csv', 'beebone_dga.csv', 'chinad_dga.csv',
        'volatilecedar_dga.csv'
    ]
    benign_ds = ['alexa_top_1m.csv', 'majestic_million.csv']
    # setup paths
    data_path = repo_path / "data"
    raw_path = data_path / "raw"
    # collect domains and labels
    dga_df = get_dga_data(raw_path, dga_ds, n=n_dga)
    ben_df = get_benign_data(raw_path, benign_ds, n=n_ben)
    df = pd.concat([dga_df, ben_df])
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # save dataset
    prepared = data_path / "processed"
    save_as_csv(df, prepared / ds_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='dataset.csv')
    parser.add_argument('--n_ben', type=int, default=500000, metavar='N',
                        help='Number of domains from each benign dataset (default: 500000)')
    parser.add_argument('--n_dga', type=int, default=100000, metavar='N',
                        help='Number of domains from each dga dataset (default: 100000)')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data')
    args = parser.parse_args()
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path, args.ds_name, args.n_ben, args.n_dga, args.shuffle)
