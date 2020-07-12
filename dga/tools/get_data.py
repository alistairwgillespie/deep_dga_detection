"""
Little utility for parsing many csv files into one big csv,
domain data of course.
"""

import os
import csv

filenames = os.listdir("data/dga")
sep = ","


def check_data(data):
    # ... your tests
    return True  # << True if data should be written into target file, else False


with open("data/train_example.csv", "a+") as targetfile:
    writer = csv.writer(targetfile)
    for filename in filenames:
        with open("data/dga/"+filename, "r") as f:
            csv_reader = csv.reader(f, skipinitialspace=True)
            for line in csv_reader:
                data = line[0]
                if check_data(data):
                    writer.writerow([data, filename.split('.')[0]])
