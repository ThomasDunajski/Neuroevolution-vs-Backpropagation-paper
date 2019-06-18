import csv
import datetime
from utility import *

inputs = []
outputs = []

with open('training_data_normalized.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Started reading training_data_normalized.csv')  
        else:
            data = (float(row[2]),\
                float(row[3]),\
                float(row[4]),\
                float(row[5]),\
                float(row[6]),\
                float(row[7]),\
                float(row[8]),\
                float(row[9]),\
                float(row[10]))
            inputs.append(data)
            outputs.append((scaleMPG(float(row[1])),))
            #print(data)
        line_count += 1
           
    #print(f'Processed {line_count} lines.')
testInputs = []
testOutputs = []

with open('normalized_test_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Started reading normalized_test_data.csv')  
        else:
            data = (float(row[2]),\
                float(row[3]),\
                float(row[4]),\
                float(row[5]),\
                float(row[6]),\
                float(row[7]),\
                float(row[8]),\
                float(row[9]),\
                float(row[10]))
            testInputs.append(data)
            testOutputs.append((scaleMPG(float(row[1])),))
            #print(data)
        line_count += 1
           
    #print(f'Processed {line_count} lines.')
