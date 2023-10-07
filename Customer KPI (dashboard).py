# Databricks notebook source
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import List
import os 
from glob import glob
import json
import traceback
import numpy as np

from plotly_calplot import calplot 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# COMMAND ----------

customer=dbutils.widgets.get("Customer")
job_type=dbutils.widgets.get("Job type")
job_cluster=dbutils.widgets.get("Job cluster")
date_end=dbutils.widgets.get("Date End")
date_start=dbutils.widgets.get("Date Start")
(customer, job_type, job_cluster, date_end, date_start)

# COMMAND ----------

query=f'\
  select messagetimestamp, customer, element, Pass_KPIs, Fail_KPIs, Fail_list, uuid, job_type, job_cluster,  payload_path \
  from customers.fail_report\
  where messagetimestamp>="{date_start}"\
  and messagetimestamp<="{date_end}"'
if customer!="All":
  query+=f' and customer="{customer}"'
if job_type!="All":
  query+=f' and job_type="{job_type}"'
  
print(query)
df = spark.sql(query).toPandas()
df.head()

# COMMAND ----------

customer_df = spark.sql(f'select distinct customer \
                  from customers.customer_devices').toPandas()
customer_list=list(customer_df["customer"])

# COMMAND ----------

dbutils.widgets.dropdown("Customer", "All", ["All"]+customer_list)
dbutils.widgets.dropdown("Job type", "All", ["All", "Printing", "Curing"])
dbutils.widgets.dropdown("Job cluster", "All", ["All"]+[str(x) for x in df["job_cluster"].unique() if x != -1])

# COMMAND ----------

default_start=datetime.datetime.now() - datetime.timedelta(weeks=2)
default_start=default_start.strftime("%Y-%m-%d")
default_end=datetime.datetime.now() 
default_end=default_end.strftime("%Y-%m-%d")

# COMMAND ----------

dbutils.widgets.text("Date Start", default_start)
dbutils.widgets.text("Date End", default_end)

# COMMAND ----------

if customer!="All":
  df=df[df["customer"]==customer]
  if job_cluster!="All" and job_type!="Curing":
    job_cluster=int(job_cluster)
    df=df[df["job_cluster"]==job_cluster]

if job_type!="All":
  df=df[df["job_type"]==job_type]

df["messagetimestamp"]= df["messagetimestamp"].astype('datetime64[ns]')
#df.sort_values(by='messagetimestamp', inplace = True) 
df["date"] = df['messagetimestamp'].dt.strftime("%Y/%m/%d %H:%M:%S")
df.head()

# COMMAND ----------

if job_cluster!="All" and customer=="All":
  html_code="""
  <p>Warning: selecting a job cluster has no effect if you are not filering by customer</p>
  """
else:
  html_code=""
displayHTML(html_code)

# COMMAND ----------

def make_query(paths):
  params="/?databricks=true"
  for path in paths:
    new_select="&selected="+path.split("/")[-1]
    params+=new_select
  return params

base_link="http://npdu445.bcn.rd.hpicorp.net:9100"
#base_link="http://localhost:9100"
query_params=""
#query_params="/?databricks=true&selected=&selected=..."
query_params=make_query(df["payload_path"])
link=base_link+query_params

if customer!="All":
  html_code= '''<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <button><a href="'''+link+'''">See in Comparator</a></button>'''
else:
  html_code="<p>Select a customer to see jobs in Comparator</p>"
displayHTML(html_code)

# COMMAND ----------


table=df.groupby(["customer", "job_type"]).size().reset_index().rename(columns={"customer":"Customer","job_type":"Job type", 0: "Num. jobs"})
table["Job clusters"]=table.apply(lambda x: sorted(df[df["customer"]==x["Customer"]]["job_cluster"].unique()) if x["Job type"]=="Printing" else "", axis=1)
table

# COMMAND ----------

fail_kpis = []

for index, row in df.iterrows():
    fail_list, date =row["Fail_list"], row["date"]
    if len(fail_list):
      for fail_item in fail_list:
            fail_kpis.append({"fail KPI":fail_item, "date":date})
    else:
      pass

fails_df = pd.DataFrame(fail_kpis)
fails_df["date"]=fails_df["date"].astype('datetime64[ns]')

fails_df.head(n=10)

# COMMAND ----------

fails=df[["date", "Fail_KPIs", "Pass_KPIs"]]
display(fails)

# COMMAND ----------

jobs_per_day=pd.DataFrame(df['messagetimestamp'].dt.strftime("%Y-%m-%d")).groupby("messagetimestamp").size().reset_index()
start_date=min(jobs_per_day["messagetimestamp"])
end_date=datetime.datetime.now().strftime("%Y-%m-%d")
jobs_per_day["messagetimestamp"]=pd.to_datetime(jobs_per_day["messagetimestamp"], format="%Y-%m-%d")
print(start_date)
print(end_date)
jobs_per_day


# COMMAND ----------

# creating the plot
fig = calplot(
         jobs_per_day,
         x="messagetimestamp",
         y=0
)
fig.show()

# COMMAND ----------



# COMMAND ----------

pareto_df=fails_df.groupby(['fail KPI'],as_index=False).size()
pareto_df.columns=["KPI", "Num. fails"]
pareto_df=pareto_df.sort_values(by=["Num. fails"])
px.bar( pareto_df, x="KPI", y="Num. fails",title="Num. fails per KPI")

# COMMAND ----------

grouped_fails=fails_df.groupby(['date', 'fail KPI']).size().reset_index().pivot(columns='fail KPI', index='date', values=0)
grouped_fails["date"]=grouped_fails.index
grouped_fails["date"]=grouped_fails["date"].apply(lambda x : x.strftime("%m-%d-%Y %H:%M:%S"))

# COMMAND ----------

px.bar(grouped_fails, x="date", y=grouped_fails.columns)

# COMMAND ----------

if customer!="All":
  print("Retrieving ink traces")
  payload_df = spark.sql(f'select ink, job_cluster \
                  from customers.fail_report\
                  where  customer="'+customer+'" and job_type="Printing"').toPandas()
  print(payload_df.head())

# COMMAND ----------


def plot_ink_sensors(list_ink_layers, title=""):
    '''Line plot ink sensor values for every job'''
    if len(list_ink_layers)>1:
      n=len(list_ink_layers)
      fig = make_subplots(rows=1, cols=n)
      fig.update_layout(showlegend=False, height=500, width=100, title_text=title)

      for i in range(n):
        fig.add_trace(
          go.Scatter(x=[layer for layer in range(len(list_ink_layers[i]))], y=list_ink_layers[i]),
          row=1, col=i+1
        )
      fig.show()
    elif len(list_ink_layers)==1:
        plt.plot(list_ink_layers[0])

if customer!="All":
  max_column=7
  num_clusters=len(payload_df["job_cluster"].unique())
  fig = make_subplots(rows=num_clusters, cols=max_column, row_titles= [f"{i}\n" for i in payload_df["job_cluster"].unique()])
  for job_num, job in enumerate(payload_df["job_cluster"].unique()):
    print(job)
    list_ink_layers=[]

    for i, row in payload_df[payload_df["job_cluster"]==job].iterrows():
      ink_layer=row['ink']
      if ink_layer is not None:
        list_ink_layers.append(ink_layer)
        
    if len(list_ink_layers)>1:
      n=len(list_ink_layers)
      print(n)
      #fig = make_subplots(rows=1, cols=n)
      fig.update_layout(showlegend=False, height=num_clusters*250, width=1000, title_text="Job Clusters")

      for j in range(min(n, max_column)):
        fig.add_trace(
          go.Scatter(x=[layer for layer in range(len(list_ink_layers[j]))], y=list_ink_layers[j],
                     marker_color = 'rgba(0, 50, 250, .9)'),
          row=job_num+1, col=j+1
        )
    
    elif len(list_ink_layers)==1:
        plt.plot(list_ink_layers[0])
  fig.show()
else:
  html_code="<p>Select a customer to see job clusters</p>"
  displayHTML(html_code)
