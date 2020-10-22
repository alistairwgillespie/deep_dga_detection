import time
import pandas as pd
import requests

if __name__ == "__main__":
    starttime = time.time()
    c_size = 10
    n_seconds = 0
    columns_names = ['ix', 'domain', 'prediction', 'entropy']
    df = pd.DataFrame(columns=columns_names)
    df.to_csv('data/logs.csv', index=False)
    for gm_chunk in pd.read_csv('data/processed/demo_dataset.csv', chunksize=c_size):
        print("Streaming data...")
        resp = requests.post(
            "http://localhost:5000/append",
            verify=False,
            json={'data': gm_chunk['domain'].values.tolist()})
        time.sleep(1 - ((time.time() - starttime) % 1))
        n_seconds += 1
        if n_seconds == 10:
            break
    print("--- Execution time spent: %s seconds ---" % (time.time() - starttime))
