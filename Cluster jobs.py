# Databricks notebook source
# MAGIC %pip install dtw-python

# COMMAND ----------

from sklearn.cluster import dbscan
import numpy as np
from dtw import *
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#from scipy.cluster.hierarchy import linkage

# COMMAND ----------

df = spark.sql(f'select customer, messagetimestamp, payload, uuid, job_type \
                from customers.fail_report\
                where element="Printer"\
                order by uuid asc').toPandas()
df

# COMMAND ----------

#REPORTS=spark.sparkContext.textFile("/dbfs/tmp/GKNExportBronze/BuildReport_ES97R00015_202303081135.json,/dbfs/tmp/GKNExportBronze/BuildReport_ES97R00015_202304121051.json,/dbfs/tmp/GKNExportBronze/BuildReport_ES97R00023_202303081219.json").collect()
#REPORTS[0]

# COMMAND ----------

def subsample(ink, k=10):
    n=len(ink)//k

    ink_subs=np.array([np.mean(ink[k*i:k*i+k]) for i in range(n)])
    return ink_subs

def alignment_ink(ink_1, ink_2, plot=False):
    #normalize time series to [0,1]
    ink_1=(np.array(ink_1)-min(ink_1))/max(ink_1)
    ink_2=(np.array(ink_2)-min(ink_2))/max(ink_2)
    ink_1=subsample(ink_1)
    ink_2=subsample(ink_2)
    ## Align using Dynamic Time Warping 
    dtw_res=dtw(ink_1, ink_2,
        step_pattern='asymmetric', open_begin=True, open_end=True)

    ## Print plot
    if plot: dtw_res.plot(type="twoway",offset=-2)
    return dtw_res.distance

def compute_distance_matrix(list_ink_layers):
    ''' Compute distance matrix from ink layer values with DTW '''

    print("Computing distance matrix...")
    n=len(list_ink_layers)
    dist_matrix=np.empty((n,n))
    for i in range(n):
        dist_matrix[i,i]=0
        for j in range(0,i):
            ink_1=list_ink_layers[i]
            ink_2=list_ink_layers[j]
            dist_matrix[i,j]=min(alignment_ink(ink_1,ink_2),alignment_ink(ink_2,ink_1))
            dist_matrix[j,i]=dist_matrix[i,j]
    return dist_matrix

def plot_ink_sensors(list_ink_layers, title=""):
    '''Line plot ink sensor values for every job'''
    n=len(list_ink_layers)
    fig, axs = plt.subplots(1, n, figsize=(100,5))
    fig.suptitle(title, fontsize=50)
    for i in range(n):
        ink_layer=list_ink_layers[i]
        axs[i].plot(ink_layer)

def subsample(ink, k=10):
    n=len(ink)//k

    ink_subs=np.array([np.mean(ink[k*i:k*i+k]) for i in range(n)])
    return ink_subs

def alignment_ink(ink_1, ink_2, plot=False):
    #normalize time series to [0,1]
    ink_1=(np.array(ink_1)-min(ink_1))/max(ink_1)
    ink_2=(np.array(ink_2)-min(ink_2))/max(ink_2)
    ink_1=subsample(ink_1)
    ink_2=subsample(ink_2)
    ## Align using Dynamic Time Warping 
    dtw_res=dtw(ink_1, ink_2,
        step_pattern='asymmetric', open_begin=True, open_end=True)

    ## Print plot
    if plot: dtw_res.plot(type="twoway",offset=-2)
    return dtw_res.distance

def compute_distance_matrix(list_ink_layers_uuid):
    ''' Compute distance matrix from ink layer values with DTW '''
    list_ink_layers=[l[0] for l in list_ink_layers_uuid]
    print("Computing distance matrix...")
    n=len(list_ink_layers)
    dist_matrix=np.empty((n,n))
    for i in range(n):
        dist_matrix[i,i]=0
        for j in range(0,i):
            ink_1=list_ink_layers[i]
            ink_2=list_ink_layers[j]
            dist_matrix[i,j]=min(alignment_ink(ink_1,ink_2),alignment_ink(ink_2,ink_1))
            dist_matrix[j,i]=dist_matrix[i,j]
    return dist_matrix

def plot_ink_sensors(list_ink_layers_uuid, title=""):
    '''Line plot ink sensor values for every job'''
    list_ink_layers=[l[0] for l in list_ink_layers_uuid]
    if len(list_ink_layers)>1:
        n=len(list_ink_layers)
        fig, axs = plt.subplots(1, n, figsize=(100,5))
        fig.suptitle(title, fontsize=50)
        for i in range(n):
            ink_layer=list_ink_layers[i]
            axs[i].plot(ink_layer)
    elif len(list_ink_layers)==1:
        plt.plot(list_ink_layers[0])


customer_dist_matrices={}
customer_list_ink={}
customer_list=list(df["customer"].unique())
for customer in customer_list:
    print(f"Customer: {customer}")
    list_ink_layers=[]
    for i, row in df[df["customer"]==customer].iterrows():
        payload=json.loads(row['payload'])
        contexts=payload["payload"]['sensorsData']
        for context in contexts:
            for sensor in context["sensors"]:
                if sensor["name"]=="Ink":
                    ink_layer=sensor["layerData"]
                    if len(ink_layer)>400:
                        list_ink_layers.append((ink_layer, row["uuid"], row["messagetimestamp"]))
    customer_list_ink[customer]=list_ink_layers
    n=len(list_ink_layers)
    # Plot ink sensor values for every job
    # if n<15: plot_ink_sensors(list_ink_layers)

    # Compute and plot distance matrix
    dist_matrix=compute_distance_matrix(list_ink_layers)
    customer_dist_matrices[customer]=dist_matrix

# COMMAND ----------

s=set()
s.add((1,2))
s.add((3,2))

(2,2) in s

# COMMAND ----------

customer = customer_list[4]
print(customer)
dist_matrix=customer_dist_matrices[customer]
list_ink_layers=customer_list_ink[customer]
print(np.shape(dist_matrix))
plt.hist(dist_matrix.reshape(-1), bins="auto")
plt.show()
plot_ink_sensors(list_ink_layers)
plt.matshow(dist_matrix,cmap="Reds")
clustering = DBSCAN(eps=2,metric="precomputed", min_samples=2).fit(dist_matrix)
print(clustering.labels_)

# COMMAND ----------

uuids=[l[1] for l in list_ink_layers]
mapping={}
for uuid, label in zip(uuids, clustering.labels_):
    print(uuid, label)
    mapping[uuid]=label 
mapping

# COMMAND ----------

labels=set(clustering.labels_)
values=list(zip(clustering.labels_, list_ink_layers))

groupped=[]
for x in labels:
  group=[]
  for label, list_ink in values:
    if label==x and label!=-1:
      group.append(list_ink)
  groupped.append((group, x))

for group, x in groupped:
  if len(group):
    print(len(group))
    plot_ink_sensors(group, title=f"Job type {x}")
