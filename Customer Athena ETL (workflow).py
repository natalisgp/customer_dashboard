# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is to be run every night to extract the data from customers periodically. 
# MAGIC - This notebook needs that the `customer_devices` table is updated every time a new machine is deployed.
# MAGIC - This notebook needs to be run in a cluster where the AEoTv2_SDK library is installed (e.g. 3D-R&d generic Cluster or WS tools Cluster)
# MAGIC

# COMMAND ----------

exportFolder = "/dbfs/tmp/CustomerExportBronze/"

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /var/log/aeot

# COMMAND ----------

dbutils.fs.cp('dbfs:/dbfs/FileStore/AEoTConfig/config.json', 'file:/home/aeot/config.json')

# COMMAND ----------

from AEoTv2_SDK.AthenaService.AthenaService import AthenaService, AthenaResponseSchema
from AEoTv2_SDK.Logger.LoggerService import LoggerService
from AEoTv2_SDK.Configurations.ConfigurationsService import ConfigurationsService
import pandas as pd
from pandas import DataFrame
from typing import List
import os 
from glob import glob
import json
import traceback
import datetime as dt

import matplotlib.pyplot as plt
from dtw import *
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

# COMMAND ----------


def obtain_level_1( report_file,debug=False ):

    #f = open(report_file)
    #data = json.load(f)
    data = json.loads(report_file)
    df_output = pd.DataFrame()
    
    ok_count = 0
    fail_count = 0
    fail_list = []

    #Checking Sensors
    try:
        for context_index in range(len(data["payload"]["sensorsData"])):
            
            if (debug):
                print ("\tcontext->",data["payload"]["sensorsData"][context_index]["context"])
                print ("\tNumber of sensors ->",len(data["payload"]["sensorsData"][context_index]["sensors"]) )
                print ("\tFields->",data["payload"]["sensorsData"][context_index]["sensors"][0].keys())
                
            context = data["payload"]["sensorsData"][context_index]["context"]
            
            for sensor_index in range(len(data["payload"]["sensorsData"][context_index]["sensors"])):
                average = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["average"]
                max = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["max"]
                min = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["min"]
                name = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["name"]
                range_lower = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["range"]["lower"]
                range_upper = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["range"]["upper"]
                units = data["payload"]["sensorsData"][context_index]["sensors"][sensor_index]["units"]

                delta = 0
                # Special case for layer time:
                if name in ["Layers Duration"]:
                    delta = average

                range_min = range_lower + delta
                range_max = range_upper + delta
                status = range_min <= min <= max <= range_max 
                if status:
                    ok_count = ok_count + 1
                else:
                    fail_count = fail_count + 1
                    fail_list.append(name)


    except Exception as e:
            print(f"Exception parsing Sensors: Exception({e})")
            print(traceback.format_exc())


    #Checking Servos
    try:
        for context_index in range(len(data["payload"]["servosData"])):
            
            if (debug):
                print ("\tcontext->",data["payload"]["servosData"][context_index]["context"])
                print ("\tNumber of sensors ->",len(data["payload"]["servosData"][context_index]["servos"]) )
                print ("\tFields->",data["payload"]["servosData"][context_index]["servos"][0].keys())
                print ("\n")
            
            context = data["payload"]["servosData"][context_index]["context"]
            
            for servo_index in range(len(data["payload"]["servosData"][context_index]["servos"])):
                deltaAverage = data["payload"]["servosData"][context_index]["servos"][servo_index]["deltaAverage"]
                deltaMax = data["payload"]["servosData"][context_index]["servos"][servo_index]["deltaMax"]
                deltaMin = data["payload"]["servosData"][context_index]["servos"][servo_index]["deltaMin"]
                deltaStdev = data["payload"]["servosData"][context_index]["servos"][servo_index]["deltaStdev"]
                name = data["payload"]["servosData"][context_index]["servos"][servo_index]["name"]
                pwmAverage = data["payload"]["servosData"][context_index]["servos"][servo_index]["pwmAverage"]
                targetAverage = data["payload"]["servosData"][context_index]["servos"][servo_index]["targetAverage"]
                range_lower = data["payload"]["servosData"][context_index]["servos"][servo_index]["range"]["lower"]
                range_upper = data["payload"]["servosData"][context_index]["servos"][servo_index]["range"]["upper"]
                units = data["payload"]["servosData"][context_index]["servos"][servo_index]["units"]
                
                status = range_lower <= deltaMin <= deltaMax <= range_upper

                if status:
                    ok_count = ok_count + 1
                else:
                    fail_count = fail_count + 1
                    fail_list.append(name)

    except Exception as e:
        print(f"Exception parsing Sensors: Exception({e})")
        print(traceback.format_exc())


    #Checking singleData 
    #TODO:pending to implement
    try:
        if (debug):
            print ("\tNumber of singleData ->",len(data["payload"]["singleValueData"]) )
            print ("\tFields->",data["payload"]["singleValueData"][0].keys())
            print ("\n")
    except Exception as e:
            print(f"Exception parsing Sensors: Exception({e})")
            print(traceback.format_exc())

    return (ok_count,fail_count,fail_list)

# COMMAND ----------

def create_dataframe_from_query_results(query_results: List) -> DataFrame:
    if not query_results:
        return pd.DataFrame([], columns = [])
    row_values = list()
    column_names = query_results[0].keys()
    for row in query_results:
        row_values.append(row.values())
    return pd.DataFrame(row_values, columns = column_names)

# COMMAND ----------

def identify_job_type(report):
  report=str(report.get("payload").get("sensorsData"))+str(report.get("payload").get("servosData"))
  if "Unpacking" in report or "TrolleyCleaning" in report:
    return "Depowder"
  elif "Building|Warmup" in report or "Building|Printing" in report:
    return "Printing"
  elif "MaterialLoading" in report:
    return "Loading"
  elif "Curing" in report:
    return "Curing"

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

def compute_distance_matrix(list_ink_layers_uuid, customer):
    ''' Compute distance matrix from ink layer values with DTW '''
    print("Computing distance matrix...")
    n=len(list_ink_layers_uuid)
    dist_matrix=np.empty((n,n))
    rows=[]
    for i in range(n):
        dist_matrix[i,i]=0
        for j in range(0,i):
            row={"customer":customer, "job1":list_ink_layers_uuid[i][1], "job2":list_ink_layers_uuid[j][1]}
            ink_1=list_ink_layers_uuid[i][0]
            ink_2=list_ink_layers_uuid[j][0]
            dist_matrix[i,j]=min(alignment_ink(ink_1,ink_2),alignment_ink(ink_2,ink_1))
            dist_matrix[j,i]=dist_matrix[i,j]
            row["dist"]=dist_matrix[i,j]
            rows.append(row)
    return dist_matrix, pd.DataFrame(rows)

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

def cluster_jobs(df):
  customer_dist_matrices={}
  customer_list_ink={}
  customer_list=list(df["customer"].unique())
  all_dist_df=[]
  for customer in customer_list:
    try:
      print(f"Customer: {customer}")
      list_ink_layers=[]
      for i, row in df[df["customer"]==customer].iterrows():
        ink_layer=row["ink"]
        if ink_layer:                
          list_ink_layers.append((ink_layer, row["uuid"], row["messagetimestamp"]))
      customer_list_ink[customer]=list_ink_layers
      n=len(list_ink_layers)
      print(n)

      dist_matrix, dist_df=compute_distance_matrix(list_ink_layers, customer)
      customer_dist_matrices[customer]=dist_matrix
      all_dist_df.append(dist_df)
    except Exception as e:
      print(str(e))
  dist_df=pd.concat(all_dist_df).head()

  # Save in table
  spark_df = spark.createDataFrame(dist_df)
  spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("hive_metastore.customers.job_distances")

  return customer_dist_matrices
#customer_dist_matrices=save_distance_matrix(df)

# COMMAND ----------

def update_job(x, map_to_type):
  key=(x["uuid"],x["messagetimestamp"])
  
  if key in map_to_type.keys():
    return map_to_type[key]
  return -1

def cluster_jobs(customer_dist_matrices, df):
  customer_list=customer_dist_matrices.keys()
  for customer in customer_list:
    print(customer)
    try:
      dist_matrix=customer_dist_matrices[customer]
      list_ink_layers=customer_list_ink[customer]
      clustering = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05, algorithm="brute").fit(dist_matrix)
      analyzed=[(uuid, time) for ink, uuid, time in list_ink_layers]
      map_to_type={key: value for key, value in zip(analyzed, clustering.labels_)}
      df.loc[df["customer"]==customer,"job_cluster"]=df[df["customer"]==customer].apply(lambda x: update_job(x, map_to_type), axis=1)
    except Exception as e:
      print(str(e))
  return df


def extractJSON(row):
  '''Returns the full path where the report is stored and 
  whether or not the report has been stored previously'''
  pl = row['payload'];
  #filename = "BuildReport"+"_"+row['deviceserialnumber']+"_"+str(row['messageyear'])+str(row['messagemonth']).zfill(2)+str(row['messageday']).zfill(2)
  filename = "BuildReport"+"_"+row['deviceserialnumber']+"_"+row['messagetimestamp'].strftime("%Y%m%d%H%M%S")+".json"
  #print("Report-:",row['messageyear'],row['messagemonth'],row['messageday'],row['deviceserialnumber']+ pl)
  exportFolder = "/dbfs/tmp/CustomerExportBronze/"
  report_names=[file.name for file in dbutils.fs.ls(exportFolder)]
  if filename not in report_names:
    print("Report name:",exportFolder+filename)

    dbutils.fs.rm(exportFolder+filename)
    dbutils.fs.put(exportFolder+filename,pl)
    return (exportFolder+filename, 0)
  else:
    print("DUPLICATE")
    return (exportFolder+filename, 1)

def extractInk(payload_str):
  '''Given the report in string format it returns a list of
  ink traces for the job if there is ink data in the report'''

  payload=json.loads(payload_str)
  contexts=payload["payload"]['sensorsData']
  for context in contexts:
      for sensor in context["sensors"]:
          if sensor["name"]=="Ink":
              return sensor["layerData"]
  return None

# Loop to extract payloads for analysis
def extractJSONRow(row):
  return(row['payload'])

# Extract uuid rom row's payload
def extractUUID(x):
  try:
    return json.loads(x["payload"])["payload"]["actualBuildTicket"]["jobUuid"]
  except Exception as e:
    print(e)
    return None
#df=cluster_jobs(customer_dist_matrices, df)

# COMMAND ----------

def update_db(start, end):
  config = ConfigurationsService()
  logger = LoggerService()
  athena_service = AthenaService(config, logger, stage="prod")
  #athena_service = AthenaService(config, logger, stage="stage")

  df_devices = spark.sql(f"select Customer, SN as deviceserialnumber, Element\
                           from customers.customer_devices \
                           where Element!='Build Unit' ").toPandas()
  serials = '('+','.join(map(lambda x : "'"+str(x)+"'", list(df_devices["deviceserialnumber"])))+')'

  query_prod = f"""
      SELECT 
        messagetimestamp
        ,deviceserialnumber
        ,messageyear
        ,messagemonth
        ,messageday
        ,payload
      FROM "3d_prod".gbd_3dp_data_lake_prod_telemetry_message_table
      WHERE
        true
        and (messageyear*10000 + messagemonth*100 + messageday) <= """+end.strftime("%Y%m%d")+"""
        and (messageyear*10000 + messagemonth*100 + messageday) >= """+start.strftime("%Y%m%d")+"""
        and deviceserialnumber not like 'HPSIM%'	-- ignore simulators
        and deviceserialnumber in """+serials+"""	-- ignore simulators
        and topic in ('Public/BuildManager/CuringReport','Public/BuildManager/BuildReport')
  """

  result_prod = athena_service.execute_query_and_get_results(query_prod)
  df = create_dataframe_from_query_results(result_prod)

  sn_to_customer={}
  sn_to_element={}
  for row in df_devices.iterrows():
    sn_to_customer[row[1]["deviceserialnumber"]] =row[1]["Customer"]
    sn_to_element[row[1]["deviceserialnumber"]] =row[1]["Element"]

  df["customer"]=df.apply(lambda x: sn_to_customer[x["deviceserialnumber"]], axis=1)
  df["element"]=df.apply(lambda x: sn_to_element[x["deviceserialnumber"]], axis=1)
  df["job_type"]=df.apply(lambda x: identify_job_type(json.loads(x["payload"])), axis=1)
  df["messagetimestamp"]=pd.to_datetime(df["messagetimestamp"],unit='us')

  errors_df = pd.DataFrame()
  fail_kpis = []
  counter = 0

  # Loop to create files
  paths=[]
  dup=[]
  inks=[]
  for index, row in df.iterrows():
      path, repeated=extractJSON(row)
      paths.append(path)
      dup.append(repeated)
      
  df["payload_path"]=paths
  df["duplicate"]=dup

  df["ink"]=df["payload"].apply(extractInk)

  #add columns to track pass and fail kpis
  df["Pass_KPIs"] = 0
  df["Fail_KPIs"] = 0
  df["Fail_list"] = df.apply(lambda x: [], axis=1)
  df["uuid"] = df.apply(extractUUID, axis=1)
  print(df)
  for index, row in df.iterrows():
    if not row['duplicate']:
      pld = extractJSONRow(row)
      (ok,fail,fail_list) = obtain_level_1(pld)
      
      date= row['messagetimestamp'].strftime("%Y/%m/%d %H:%M")
      if (fail_list):
        for fail_item in fail_list:
              fail_kpis.append({"fail KPI":fail_item, "date":date})
              counter = counter + 1
      else:
        pass
        #fail_kpis.append({"fail KPI":"", "date":date})
        #counter = counter + 1

      df.at[index,'Pass_KPIs'] = ok
      df.at[index,'Fail_KPIs'] = fail
      df.at[index,'Fail_list'] = fail_list

  df["job_cluster"]=-1
  df=df[df["uuid"]!=""]
  df_dedup=df[df["duplicate"]==0]
  write_df=df_dedup.drop(["payload", "duplicate"], axis=1)
  print(f"Detected {len(df)} jobs.Writing {len(write_df)} new rows")

  column_order=["messagetimestamp", "messagemonth",	"messageyear",	"deviceserialnumber",	"messageday",	"customer",	"element",	"job_type",	"payload_path",	"ink",	"Pass_KPIs",	"Fail_KPIs",	"Fail_list",	"uuid",	"job_cluster"]
  write_df=write_df[column_order]

  print(write_df)
  print(write_df.columns)

  if len(write_df):
    spark_df = spark.createDataFrame(write_df)
    spark_df.write.mode("append").insertInto("hive_metastore.customers.fail_report")
  
  return df, write_df

# COMMAND ----------

### Get all jobs from 6 months ago

# end=dt.datetime.now()

# while end>dt.datetime.now() - dt.timedelta(days=30*6):
#   print(start)
#   start = end - dt.timedelta(days=5)
#   df, write_df = update_db(start, end)

#   end=start

# COMMAND ----------

end=dt.datetime.now()
start=dt.datetime.now() - dt.timedelta(days=2)
df, write_df = update_db(start, end)

# COMMAND ----------

# spark_df = spark.createDataFrame(write_df)

# spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("hive_metastore.customers.fail_report")


# COMMAND ----------

# MAGIC %sql
# MAGIC --truncate table hive_metastore.customers.fail_report;
# MAGIC --select * from hive_metastore.customers.fail_report

# COMMAND ----------

# MAGIC %sql
# MAGIC -- drop table 3d_schema.buildReportGKN;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE TABLE IF NOT EXISTS 3d_schema.buildReportGKN 
# MAGIC -- USING JSON OPTIONS ("multiline" = "true" ) LOCATION "/dbfs/tmp/GKNExportBronze";
# MAGIC
# MAGIC -- CREATE TABLE IF NOT EXISTS 3d_schema.buildReportGKN_Bronze
# MAGIC -- AS SELECT *, 
# MAGIC --    input_file_name() as filename,  
# MAGIC --    split(input_file_name(), '_')[0] as topic, 
# MAGIC --    split(input_file_name(), '_')[1] as SerialNumber, 
# MAGIC --    substring(split(split(input_file_name(), '_')[2], '\\.')[0], 1,12) as job_date 
# MAGIC --    FROM 3d_schema.buildReportGKN;

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- select job_date from `3d_schema`.buildreportgkn_bronze

# COMMAND ----------

# DBTITLE 1,Create VS View
# MAGIC %sql
# MAGIC -- CREATE OR REPLACE VIEW GKN_VS_VALUES AS SELECT date_hour,  SerialNumber, buildName, JobUUID, Parameters.name as Name, CAST(Parameters.value AS INTEGER) AS Value FROM (SELECT *  FROM 
# MAGIC --       (SELECT TO_TIMESTAMP(job_date, 'yyyyMMddHHmm') as date_hour,
# MAGIC --             SerialNumber, 
# MAGIC --             payload.requestedbuildticket.jobUuid AS JobUUID, 
# MAGIC --             payload.requestedbuildticket.buildName AS buildName, 
# MAGIC --             explode(payload.requestedbuildticket.printingProfile.parameters) as Parameters 
# MAGIC --       FROM `3d_schema`.buildReportGKN_Bronze)
# MAGIC -- WHERE Parameters.name LIKE "VS-%")
