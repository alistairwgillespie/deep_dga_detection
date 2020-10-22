"""
Make dataset pipeline
"""
import pandas as pd
import numpy as np 

df = pd.read_csv("../data/processed/dns.csv")

a_aaaa_df = df.loc[(df.qtype_name == 'A') | (df.qtype_name == 'AAAA')]

# Take subset by nxdomain response
nxdomain_df = a_aaaa_df.loc[(df['rcode_name'] == 'NXDOMAIN')]

# Drop subset from full records 
a_aaaa_df = a_aaaa_df[a_aaaa_df['rcode_name'] != 'NXDOMAIN'] 

# Load known DGAs
mal_df = pd.read_csv("../data/processed/validation.csv")
mal_df = mal_df.loc[mal_df['label'] == 1]

# Inject dga domains randomly
nxdomain_df['query'] = np.random.choice(list(mal_df['domain'].values), len(nxdomain_df))

# Put dataset back together
a_aaaa_df = pd.concat([a_aaaa_df, nxdomain_df])
a_aaaa_df['domain_name'] = a_aaaa_df['query'].str.replace('www.', '')

a_aaaa_df.drop(['QR', 'AA', 'TC', 'RD', 'Z', 'answers'], axis=1, inplace=True)
a_aaaa_df.sort_values(by=['ts'])
a_aaaa_df['domain_name'].unique()
a_aaaa_df.to_csv('../data/processed/demo_dns_logs.csv', index=False)

