#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:58:40 2023

@author: marco
"""

#Simple Elbow Criterion Analysis 

# We calculate for i = 1,2,3...N number of clusters, the Elbow value:
    
import pandas as pd #always import pandas as a first step 
from plotnine import * #for the graph later on 
from sklearn.cluster import KMeans #for clustering 

pok = pd.read_excel('pok.xlsx') #the data set
aux = pok.loc[:,["Speed","Attack","Defense"]] #columns we want to isolate 
    
elbow = [] #make an empty list to store inertia values (intertia is the value of decrease on the graph)
for n in range(1,15): #in general the range is subjective, but 1-15 is usually good enough 
    km = KMeans(n_clusters=n)
    km.fit(aux)
    elbow.append(km.inertia_) #store the intertia values into the empty list 
    
sol = pd.DataFrame({"Cluster":range(1,15),"Elbow":elbow}) #creates a df with two columns, cluster and elbow
ggplot(aes(x="Cluster", y = "Elbow"),sol) + geom_point(color="red", size=4) + geom_line() #plot the elbow values to be able to identify the 'elbow' 
