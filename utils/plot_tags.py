"""
opens csv file with transform data
sorts it in alphabetical order of the body (tags, rig, etc)
writes to new csv file
graphs the x-z position of each tag, as well as plots the overall positions of the tags and the robot trajectory
"""

import csv
import operator
import numpy as np
import matplotlib.pyplot as plt

with open('transform10.csv') as f:
    reader = csv.reader(f)
    next(reader)

    #sorts by child frame ID, the body we care about
    sortedlist = sorted(reader, key=operator.itemgetter(4))

    #writes sorted data to new csv
    with open("parsed_10.csv", mode='w') as file:
        fileWriter = csv.writer(file, delimiter=',')
        for row in sortedlist:
            fileWriter.writerow(row)

    #creates lists of each body
    with open("parsed_10.csv") as read_file:
        read_file = csv.reader(read_file)
        #next(read_file)
        x = []
        z = []
        tag8x = []
        tag8z = []
        tag9x = []
        tag9z = []
        tag10x = []
        tag10z = []
        tag11x = []
        tag11z = []
        tag12x = []
        tag12z = []
        tag13x = []
        tag13z = []
        tag14x = []
        tag14z = []        
        
        for row in read_file:
            if row[4] == "rig":
                x.append(row[5])
                z.append(row[7])
            if row[4] == "tag_8":
                tag8x.append(row[5])
                tag8z.append(row[7])
            if row[4] == "tag_9":
                tag9x.append(row[5])
                tag9z.append(row[7])
            if row[4] == "tag_10":
                tag10x.append(row[5])
                tag10z.append(row[7])
            if row[4] == "tag_11":
                tag11x.append(row[5])
                tag11z.append(row[7])
            if row[4] == "tag_12":
                tag12x.append(row[5])
                tag12z.append(row[7])
            if row[4] == "tag_13":
                tag13x.append(row[5])
                tag13z.append(row[7])
            if row[4] == "tag_14":
                tag14x.append(row[5])
                tag14z.append(row[7])



#plots the trajectory of the robot and each tag
plt.subplot(3,3,1)
plt.plot(z,x)
plt.plot(tag8z, tag8x, '-b', linestyle='', marker='o')
plt.plot(tag9z, tag9x, '-g', linestyle='', marker='o')
plt.plot(tag10z, tag10x, '-r', linestyle='', marker='o')
plt.plot(tag11z, tag11x, '-c', linestyle='', marker='o')
plt.plot(tag12z, tag12x, '-m', linestyle='', marker='o')
plt.plot(tag13z, tag13x, '-y', linestyle='', marker='o')
plt.plot(tag14z, tag14x, '-k', linestyle='', marker='o')
plt.title("Whole Map")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

#plots z-x position of each tag
#include linestyle='' to see points instead of line with points
#line with points can show link between the points over time as the robot moves

plt.subplot(3,3,2)
#plt.plot(tag8z, tag8x, '-b', linestyle='', marker='o')
plt.plot(tag8z, tag8x, '-b', marker='o')
plt.title("Tag 8")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,3)
#plt.plot(tag9z, tag9x, '-g', linestyle='', marker='o')
plt.plot(tag9z, tag9x, '-g', marker='o')
plt.title("Tag 9")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,4)
#plt.plot(tag14z, tag14x, '-k', linestyle='', marker='o')
plt.plot(tag14z, tag14x, '-k', marker='o')
plt.title("Tag 14")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,5)
#plt.plot(tag10z, tag10x, '-r', linestyle='', marker='o')
plt.plot(tag10z, tag10x, '-r', marker='o')
plt.title("Tag 10")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,6)
#plt.plot(tag12z, tag12x, '-m', linestyle='', marker='o')
plt.plot(tag12z, tag12x, '-m', marker='o')
plt.title("Tag 12")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,7)
#plt.plot(tag11z, tag11x,'-c', linestyle='', marker='o')
plt.plot(tag11z, tag11x,'-c', marker='o')
plt.title("Tag 11")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")

plt.subplot(3,3,8)
#plt.plot(tag13z, tag13x, '-y', linestyle='', marker='o')
plt.plot(tag13z, tag13x, '-y', marker='o')
plt.title("Tag 13")
plt.xlabel("Z Position (m)")
plt.ylabel("X Position (m)")


plt.show()