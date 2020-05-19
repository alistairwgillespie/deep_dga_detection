import gzip
import csv 
import re
import os 

cwd = os.getcwd()
print(cwd)
with open('/mnt/c/Users/gilleal/PycharmProjects/dga_detection/data/dns.csv', 'w') as out_file, gzip.open('/mnt/c/Users/gilleal/PycharmProjects/dga_detection/data/dns.log.gz', 'r') as in_file:
    writer = csv.writer(out_file)
    writer.writerow(['ts', 'uid', 'id.orig_h', 'id.orig_p', 'proto', 'port', 'query', 'qclass', 'qclass_name', 'qtype', 'qtype_name', 'rcode', 'rcode_name', 'QR', 'AA', 'TC', 'RD', 'Z', 'answers', 'TTLs', 'rejected'])
    for line in in_file:
        columns = re.split(b'\t+', line)
        writer.writerow([i.decode("utf-8") for i in columns])
