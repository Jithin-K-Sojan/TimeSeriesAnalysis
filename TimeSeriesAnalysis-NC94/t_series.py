import numpy as np;
from math import floor
import matplotlib.pyplot as plt;
#import sklearn as sk;
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
from tslearn.utils import to_time_series
from tslearn.clustering import TimeSeriesKMeans
from math import sqrt
from math import exp

#Time Series Clustering. Into 10 clusters

max1 = "17001"
county = 0
county1 = -1
i = int(0)
count  = int(0)
#sum1 = [0.0,0.0,0.0,0.0]
length  = ((55141-17001)//2)+1  #All counties are odd in identification number
my_array = np.empty((length,365*3,4))

#maybe year can be a span of ten years between 1899 and 1999
with open("/home/jithin_sk/Desktop/data/Crop/Climate.txt") as fileobject:
    for line in fileobject:
        S = line.split(',')
        if int(S[3])<=1999 and int(S[3])>=1997:
            index = int(S[3])-1997
            #print(int(S[0][1:6]))
            county = int(S[0][1:6]) - int(max1)
            county = county//2    
            if county1 == county:
                i+=1
                count = count+1
            else:
                '''for i in range(0,4):
                    my_array[county,i] = sum1[i]/count
                my_array[county] = sum[]
                for i in range(0,4):
                    sum1[i] = 0.0'''
                i = 0
                count = 1
                county1 = county
            #for i in range(0,4):
            #    sum1[i]+= float(S[5+i])
            if(i>=365):
                pass #leaving the one even county of 55078
            else:
                for j in range(4):
                    my_array[county,index*365+i,j] = float(S[5+j])

'''kmeans = KMeans(n_clusters=4)
kmeans.fit(my_array)
#y_kmeans = kmeans.predict(my_array)
#plt.scatter(my_array[:,0],my_array[:,3],c = y_kmeans, s=50, cmap='viridis')
#plt.show()
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
#print(my_array[500,3])
print(centroids)
print(len(labels))'''

no_clust = 10
t_series = to_time_series(my_array)
kmeans = TimeSeriesKMeans(n_clusters=no_clust,metric="euclidean",max_iter=8,random_state=0)
kmeans.fit(t_series)
print("The cluster centers are:",kmeans.cluster_centers_)
print("Each time series belongs to:",kmeans.labels_)
labels = kmeans.labels_

y_kmeans = kmeans.predict(t_series)
plt.scatter(t_series[:,0,1],[2 for _ in range(length)],c = y_kmeans, s=30, cmap='viridis')
plt.scatter(t_series[:,182,1],[1.5 for _ in range(length)],c = y_kmeans, s=30, cmap='viridis')
plt.scatter(t_series[:,364,1],[1 for _ in range(length)],c = y_kmeans, s=30, cmap='viridis')
plt.show()

plt.scatter([i for i in range(3*365)],t_series[0,:,3],s=30) #3 years worth of data
plt.show()

count_arr = [0 for _ in range(no_clust)]
for i in range(length):
    count_arr[labels[i]]+=1
for i in range(no_clust):
    print("Number of points in cluster # ",i," = ",count_arr[i])

#exit(0)

#Find sum of crop yield for say Wheat all year for each year from each of the clusters and then apply regression
#Timeseries Regression - Rolling Window

arr_prod = np.empty((no_clust,32))    #32 years worth of data

for j in range(no_clust):
    for i in range(32):
        arr_prod[j,i] = int(0)

with open("/home/jithin_sk/Desktop/data/Crop/Crop.txt") as fileobject:
    for line in fileobject:
        S = line.split(',')
        if(int(S[0][1:6])%2!=0 and int(S[0][1:6])<=55141 and int(S[0][1:6])>=17001):    #only using values that are there in the db for clustering
            if S[7]=="\"Wheat  All\"":
                #print("Hi2") 
                if S[8]=="\"Total For Crop\"":
                    #print("Hi3")
                    print(S[0][1:6])
                    county = int(S[0][1:6]) - int(max1)
                    county = county//2
                    clust = labels[county] 
                    index = int(S[6])-1970
                    if(S[17]!=""):
                        arr_prod[clust,index] += int(float(S[17]))

'''for i in range(no_clust):
    for j in range(32):
        print(arr_prod[i,j])
'''
#rolling-window mean
#Let w be the window size
alpha = 0.9
w = 3
pred1 = []
for k in range(no_clust):
    arr_pred = np.empty(32)
    train = [arr_prod[k,i] for i in range(0,w)]
    #length_test = 32-w
    for i in range(w,32):
        pred = np.mean(arr_prod[k,i-w:i])
        arr_pred[i] = pred

    for i in range(w+1,32):
        arr_pred[i] = alpha*arr_pred[i] + (1-alpha)*arr_pred[i-1]
    
    if(k==7):       #Plotting the predictedd values of the 7th cluster
        pred1 = [arr_pred[l] for l in range(0,32)]

    error = sqrt(mean_squared_error(arr_pred[w:32],arr_prod[k,w:32]))
    print("Rolling window regression with function as mean:")
    print("Window size:",w)
    print("Root-mean-squared-error(for the cluster",k,"):",error)

plt.plot([i for i in range(w,32)],pred1[w:32])
plt.scatter([i for i in range(w,32)],arr_prod[7,w:32])
plt.show()

#rolling window median
alpha = 0.9
pred1 = []
for k in range(no_clust):
    w = 3
    arr_pred = np.empty(32)
    train = [arr_prod[k,i] for i in range(0,w)]
    #length_test = 32-w
    totsum = 0
    for i in range(w,32):
        pred = np.median(arr_prod[k,i-w:i])
        arr_pred[i] = pred

    for i in range(w+1,32):
        arr_pred[i] = alpha*arr_pred[i] + (1-alpha)*arr_pred[i-1]
         
    if(k==7):           #For plotting the predited values of the 7th cluster
        pred1 = [arr_pred[l] for l in range(0,32)]

    error = sqrt(mean_squared_error(arr_pred[w:32],arr_prod[k,w:32]))
    print("Rolling window regression with function as median:")
    print("Window size:",w)
    print("Root-mean-squared-error(for the cluster",k,"):",error)

plt.plot([i for i in range(w,32)],pred1[w:32])
plt.scatter([i for i in range(w,32)],arr_prod[7,w:32])
plt.show()

#exit(0)

#Classification - As a result of regression

correct = 0
total = 0
for k in range(no_clust):

    arr_pred = np.empty(32)
    train = [arr_prod[k,i] for i in range(0,w)]
    for i in range(w,32):
        pred = np.median(arr_prod[k,i-w:i])
        arr_pred[i] = pred
    
    '''max1 = -1
    min1 =  arr_pred[w]
    for i in range(w,32):
        if(max1<arr_pred[i]):
            max1 = arr_pred[i]
        if(min1>arr_pred[i]):
            min1 = arr_pred[i]'''
    max2 = -1
    min2 = arr_prod[k,w]
    for i in range(w,32):
        if(max2<arr_prod[k,i]):
            max2 = arr_prod[k,i]
        if(min2>arr_prod[k,i]):
            min2 = arr_prod[k,i]

    for i in range(w,32):
        predNorm = (arr_pred[i]-min2)/(max2-min2)
        prodNorm = (arr_prod[k,i]-min2)/(max2-min2)
        predNorm/=0.33
        prodNorm/=0.33
        print("Cluster Number:",k)
        print("Year:",w+1970)
        print("Predicted Class:",floor(predNorm))
        print("Actual Clas:",floor(prodNorm))
        if(floor(predNorm)==floor(prodNorm)):
            correct +=1
        total+=1

print("Percentage accuracy of classification:",(correct/total)*100)