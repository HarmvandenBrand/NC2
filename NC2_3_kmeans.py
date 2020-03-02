# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:18:57 2020

@author: admin
"""
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np

def maxColumn(my_list):

    m = len(my_list)
    n = len(my_list[0])

    list2 = []  # stores the column wise maximas
    for col in range(n):  # iterate over all columns
        col_max = my_list[0][col]  # assume the first element of the column(the top most) is the maximum
        for row in range(1, m):  # iterate over the column(top to down)

            col_max = max(col_max, my_list[row][col]) 

        list2.append(col_max)
    return list2

def minColumn(my_list):

    m = len(my_list)
    n = len(my_list[0])

    list2 = []  # stores the column wise maximas
    for col in range(n):  # iterate over all columns
        col_max = my_list[0][col]  # assume the first element of the column(the top most) is the maximum
        for row in range(1, m):  # iterate over the column(top to down)

            col_max = min(col_max, my_list[row][col]) 

        list2.append(col_max)
    return list2


#artificial dataset 1
x_ai1 = np.random.uniform(low = -1, high = 1, size = (400,2))
y_ai1 = []
for i in x_ai1:
    if (i[0] >= 0.7 or (i[0] <= 0.3 and (i[1] >= -0.2-i[0]))):
        y_ai1.append(1)
    else:
        y_ai1.append(0)

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
max_iris = maxColumn(iris_x)
min_iris = minColumn(iris_x)

def calcScoreIris (assignments, labels):
    best = 0
    n = len(assignments)
    opties = 6
    for o in range(opties):
        temp = 0 
        for i in range(n):
            if o == 0:
                if assignments[i] == labels[i]:
                    temp += 1
            if o == 1:
                if ((assignments[i] == 0 and labels[i] == 0) or
                (assignments[i] == 2 and labels[i] == 1) or 
                (assignments[i] == 1 and labels[i] == 2)):
                    temp += 1
            if o == 2:
                if ((assignments[i] == 1 and labels[i] == 0) or
                (assignments[i] == 0 and labels[i] == 1) or 
                (assignments[i] == 2 and labels[i] == 2)):
                    temp += 1
            if o == 3:
                if ((assignments[i] == 1 and labels[i] == 0) or
                (assignments[i] == 2 and labels[i] == 1) or 
                (assignments[i] == 0 and labels[i] == 2)):
                    temp += 1
            if o == 4:
                if ((assignments[i] == 2 and labels[i] == 0) or
                (assignments[i] == 0 and labels[i] == 1) or 
                (assignments[i] == 1 and labels[i] == 2)):
                    temp += 1
            if o == 5:
                if ((assignments[i] == 2 and labels[i] == 0) or
                (assignments[i] == 1 and labels[i] == 1) or 
                (assignments[i] == 0 and labels[i] == 2)):
                    temp += 1
            if temp > best:
                best = temp
    return best

def calcScoreAI(assignments, labels):
    best = 0
    opties = 2
    n = len(assignments)
    for o in range(opties):
        temp = 0
        for i in range(n):
            if o == 0:
                if assignments[i] == labels[i]:
                    temp += 1
            if o == 1:
                if ((assignments[i] == 1 and labels[i] == 0) or
                (assignments[i] == 0 and labels[i] == 1)):
                    temp += 1
            if temp > best:
                best = temp
    return best

# k-means on iris
epoch = 20
number_clusters = 3

def kmeansIris (epoch, number_clusters):
    centroids = []
    scores = []
    for i in range(number_clusters):
        centroids.append([np.random.uniform(min_iris[0],max_iris[0]),
                          np.random.uniform(min_iris[1],max_iris[1]),
                          np.random.uniform(min_iris[2],max_iris[2]),
                          np.random.uniform(min_iris[3],max_iris[3])])
    for g in range(epoch):
        assignments = np.zeros(150)
        quantization = 0
        for i,j in enumerate(iris_x):
            distances = np.zeros(number_clusters)
            for d,c in enumerate(centroids):
                dist = np.sqrt(np.sum(np.square((c - j))))
                distances[d] = dist
            assign = np.where(distances == np.amin(distances))[0]
            quantization += np.amin(distances)
            assignments[i] = assign
        quantization = quantization / 150
        verdeling = np.zeros(number_clusters)
        for i in range(number_clusters):
            num_in_centroid = (assignments == i).sum()
            verdeling[i] = num_in_centroid
            som = np.zeros(4)
            for x in range(len(iris_x)):
                if (int(assignments[x]) == i):
                    som = np.add(som, iris_x[x])
            for x in range(4):
                if(num_in_centroid != 0):
                    centroids[i][x] = (1/num_in_centroid) * som[x]
        scores.append(calcScoreIris(assignments, iris_y)) 
    bestscore = 0
    besttime = 0
    for i in range(len(scores)):
        if scores[i] > bestscore:
            besttime = i
            bestscore = scores[i]
    return scores, besttime, bestscore, quantization
        
def kmeansAI (epoch, number_clusters):
    centroids = []
    scores = []
    for i in range(number_clusters):
        centroids.append([np.random.uniform(-1,1), np.random.uniform(-1,1)])
    for g in range(epoch):
        assignments = np.zeros(400)
        quantization = 0
        for i,j in enumerate(x_ai1):
            distances = np.zeros(number_clusters)
            for d,c in enumerate(centroids):
                dist = np.sqrt(np.sum(np.square((c - j))))
                distances[d] = dist
            assign = np.where(distances == np.amin(distances))[0]
            quantization += np.amin(distances)
            assignments[i] = assign
        quantization = quantization / 400
        verdeling = np.zeros(2)
        for i in range(number_clusters):
            num_in_centroid = (assignments == i).sum()
            verdeling[i] = num_in_centroid
            som = np.zeros(2)
            for x in range(len(x_ai1)):
                if (int(assignments[x]) == i):
                    som = np.add(som, x_ai1[x])
            for x in range(2):
                if(num_in_centroid != 0):
                    centroids[i][x] = (1/num_in_centroid) * som[x]
        scores.append(calcScoreAI(assignments,y_ai1)) 
    bestscore = 0
    besttime = 0
    for i in range(len(scores)):
        if scores[i] > bestscore:
            besttime = i
            bestscore = scores[i]
    return scores, besttime, bestscore, quantization

ai_quants = []
iris_quants = []
for i in range(30):
    print(i)
    ai_scores, ai_besttime, ai_bestscore, ai_quantization = kmeansAI(epoch, 2)
    iris_scores, iris_besttime, iris_bestscore, iris_quantization = kmeansIris(epoch, 3)
    ai_quants.append(ai_quantization)
    iris_quants.append(iris_quantization)
ai_mean = np.mean(ai_quants)
iris_mean = np.mean(iris_quants)
ai_std = np.std(ai_quants)
iris_std = np.std(iris_quants)
