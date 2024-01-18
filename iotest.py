import csv
import pprint

with open('iotest.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        print(len(row))




