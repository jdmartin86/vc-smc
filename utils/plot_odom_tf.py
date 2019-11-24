##organizes transform csv file by tags alphabetically and write to new "parsed_#" csv file

import pandas as pd
import csv
import operator
import numpy as np
import matplotlib.pyplot as plt

with open('transforms/csv_files/transform7.csv') as f:
    reader = csv.reader(f)
    next(reader)
    #dataframe = pd.read_csv(f)
    #print(dataframe.head(10))

    #sorted_bodies = dataframe.sort_values(by=["field.transforms0.child_frame_id", "field.transforms0.header.stamp"], ascending=True, inplace=True)
    #print(dataframe.head(5))

    sortedlist = sorted(reader, key=operator.itemgetter(4))

    with open("transforms/csv_files/parsed_.csv", mode='w') as file:
        fileWriter = csv.writer(file, delimiter=',')
        for row in sortedlist:
            fileWriter.writerow(row)

    with open("transforms/csv_files/parsed_7.csv") as read_file:
        read_file = csv.reader(read_file)
        #next(read_file)
        x1 = []
        z1 = []
        for row in read_file:
            x1.append(row[5])
            z1.append(row[7])

with open('odoms/csv_files/odom7.csv') as f:
    reader = csv.reader(f)
    next(reader)

    x2 = []
    z2 = []
    for row in reader:
        x2.append(row[5])
        z2.append(row[7])
    
#plt.subplot(1,2,1)
plt.plot(z1,x1, '-b', label='transforms', linestyle='', marker='o')

#plt.subplot(2,1,2)
plt.plot(z2,x2, '-r', label='odom', linestyle='', marker='o')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('z position (m)')
plt.ylabel('x position (m)')

plt.show()