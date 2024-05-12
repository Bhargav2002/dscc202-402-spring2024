# Databricks notebook source
# MAGIC %md
# MAGIC ## DSCC202-402 Data Science at Scale Final Project
# MAGIC ### Tracking Tweet sentiment at scale using a pretrained transformer (classifier)
# MAGIC <p>Consider the following illustration of the end to end system that you will be building.  Each student should do their own work.  The project will demonstrate your understanding of Spark Streaming, the medalion data architecture using Delta Lake, Spark Inference at Scale using an MLflow packaged model as well as Exploritory Data Analysis and System Tracking and Monitoring.</p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/pipeline.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You will be pulling an updated copy of the course GitHub repositiory: <a href="https://github.com/lpalum/dscc202-402-spring2024">The Repo</a>.  If you are unclear on how to pull an updated copy using the GitHub command line, the following <a href="https://techwritingmatters.com/how-to-update-your-forked-repository-on-github">document</a> is helpful.  Be sure to add the professors and TAs as collaborators on your project. 
# MAGIC
# MAGIC - lpalum@gmail.com GitHub ID: lpalum
# MAGIC - ajay.anand@rochester.edu GitHub ID: ajayan12
# MAGIC - divyamunot1999@gmail.com GitHub ID: divyamunot
# MAGIC - ylong6@u.Rochester.edu GitHub ID: NinaLong2077
# MAGIC
# MAGIC Once you have updates your fork of the repository you should see the following template project that is resident in the final_project directory.
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/notebooks.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You can then pull your project into the Databrick Workspace using the <a href="https://www.databricks.training/step-by-step/importing-courseware-from-github/index.html">Repos</a> feature.
# MAGIC Each student is expected to submit the URL of their project on GitHub with their code checked in on the main/master branch.  This illustration highlights the branching scheme that you may use to work on your code in steps and then merge your submission into your master branch before submitting.
# MAGIC </p>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/github.drawio.png">
# MAGIC <p>
# MAGIC Work your way through this notebook which will give you the steps required to submit a complete and compliant project.  The following illustration and associated data dictionary specifies the transformations and data that you are to generate for each step in the medallion pipeline.
# MAGIC </p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/dataframes.drawio.png">
# MAGIC
# MAGIC #### Bronze Data - raw ingest
# MAGIC - date - string in the source json
# MAGIC - user - string in the source json
# MAGIC - text - tweet string in the source json
# MAGIC - sentiment - the given sentiment of the text as determined by an unknown model that is provided in the source json
# MAGIC - source_file - the path of the source json file the this row of data was read from
# MAGIC - processing_time - a timestamp of when you read this row from the source json
# MAGIC
# MAGIC #### Silver Data - Bronze Preprocessing
# MAGIC - timestamp - convert date string in the bronze data to a timestamp
# MAGIC - mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# MAGIC - cleaned_text - the bronze text data with the mentions (@username) removed.
# MAGIC - sentiment - the given sentiment that was associated with the text in the bronze table.
# MAGIC
# MAGIC #### Gold Data - Silver Table Inference
# MAGIC - timestamp - the timestamp from the silver data table rows
# MAGIC - mention - the mention from the silver data table rows
# MAGIC - cleaned_text - the cleaned_text from the silver data table rows
# MAGIC - sentiment - the given sentiment from the silver data table rows
# MAGIC - predicted_score - score out of 100 from the Hugging Face Sentiment Transformer
# MAGIC - predicted_sentiment - string representation of the sentiment
# MAGIC - sentiment_id - 0 for negative and 1 for postive associated with the given sentiment
# MAGIC - predicted_sentiment_id - 0 for negative and 1 for positive assocaited with the Hugging Face Sentiment Transformer
# MAGIC
# MAGIC #### Application Data - Gold Table Aggregation
# MAGIC - min_timestamp - the oldest timestamp on a given mention (@username)
# MAGIC - max_timestamp - the newest timestamp on a given mention (@username)
# MAGIC - mention - the user (@username) that this row pertains to.
# MAGIC - negative - total negative tweets directed at this mention (@username)
# MAGIC - neutral - total neutral tweets directed at this mention (@username)
# MAGIC - positive - total positive tweets directed at this mention (@username)
# MAGIC
# MAGIC When you are designing your approach, one of the main decisions that you will need to make is how you are going to orchestrate the streaming data processing in your pipeline.  There are several valid approaches.  First, you may choose to start the bronze_stream and let it complete (read and append all of the source data) before preceeding and starting up the silver_stream.  This approach has latency associated with it but it will allow your code to proceed in a linear fashion and process all the data by the end of your notebook execution.  Another potential approach is to start all the streams and have a "watch" method to determine when the pipeline has processed sufficient or all of the source data before stopping and displaying results.  Both of these approaches are valid and have different implications on how you will trigger your steams and how you will gate the execution of your pipeline.  Think through how you want to proceed and ask questions if you need guidance. The following references may be helpful:
# MAGIC - [Spark Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
# MAGIC - [Databricks Autoloader - Cloudfiles](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC
# MAGIC ### Be sure that your project runs end to end when *Run all* is executued on this notebook! (15 Points out of 60)

# COMMAND ----------

pip install pyspark>=2.1.0

# COMMAND ----------

# DBTITLE 1,Pull in the Includes & Utiltites
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 1,Notebook Control Widgets (maybe helpful)
"""
Adding a widget to the notebook to control the clearing of a previous run.
or stopping the active streams using routines defined in the utilities notebook
"""
dbutils.widgets.removeAll()

dbutils.widgets.dropdown("clear_previous_run", "No", ["No","Yes"])
if (getArgument("clear_previous_run") == "Yes"):
    clear_previous_run()
    print("Cleared all previous data.")

dbutils.widgets.dropdown("stop_streams", "No", ["No","Yes"])
if (getArgument("stop_streams") == "Yes"):
    stop_all_streams()
    print("Stopped all active streams.")

from delta import *
dbutils.widgets.dropdown("optimize_tables", "No", ["No","Yes"])
if (getArgument("optimize_tables") == "Yes"):
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, BRONZE_DELTA).optimize().executeCompaction()
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, SILVER_DELTA).optimize().executeCompaction()
    # Suck up those small files that we have been appending.
    DeltaTable.forPath(spark, GOLD_DELTA).optimize().executeCompaction()
    print("Optimized all of the Delta Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.0 Import your libraries here...
# MAGIC - Are your shuffle partitions consistent with your cluster and your workload?
# MAGIC - Do you have the necessary libraries to perform the required operations in the pipeline/application?

# COMMAND ----------

# Loding all the dependencies and modules to work further.
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import * #Lab2.2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import mlflow
import mlflow.spark

# Setting the number of partitions to 8 Currently
spark.conf.set("spark.sql.shuffle.partitions", 8)
spark.conf.get("spark.sql.adaptive.enabled") # Enabling AQE for dynamically changing the partitions.
print(spark.conf.get("spark.sql.shuffle.partitions"))
spark.conf.set("spark.databricks.delta.formatCheck.enabled", "false")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
spark.conf.get("spark.sql.autoBroadcastJoinThreshold")




# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Use the utility functions to ...
# MAGIC - Read the source file directory listing
# MAGIC - Count the source files (how many are there?)
# MAGIC - print the contents of one of the files

# COMMAND ----------

# MAGIC %fs ls "s3a://voc-75-databricks-data/voc_volume/"

# COMMAND ----------


df = spark.read.json("s3a://voc-75-databricks-data/voc_volume/100019.json")

df.show()


# COMMAND ----------

print(TWEET_SOURCE_PATH)

# COMMAND ----------

files = dbutils.fs.ls("s3a://voc-75-databricks-data/voc_volume/")
num_files = len(files)
print("Number of files:", num_files)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.0 Transform the Raw Data to Bronze Data using a stream
# MAGIC - define the schema for the raw data
# MAGIC - setup a read stream using cloudfiles and the source data format
# MAGIC - setup a write stream using cloudfiles to append to the bronze delta table
# MAGIC - enforce schema
# MAGIC - allow a new schema to be merged into the bronze delta table
# MAGIC - Use the defined BRONZE_CHECKPOINT and BRONZE_DELTA paths defines in the includes
# MAGIC - name your raw to bronze stream as bronze_stream
# MAGIC - transform the raw data to the bronze data using the data definition at the top of the notebook

# COMMAND ----------

#Schema for the bronze data
raw_schema = StructType([
    StructField("date", StringType(), True),
    StructField("user", StringType(), True),
    StructField("text", StringType(), True),
    StructField("sentiment", StringType(), True)
])
#Read_Stream for the brone data
bronze_stream = (spark.readStream.format("cloudFiles") 
  .option("cloudFiles.format", "json")
  .schema(raw_schema) 
  .load(TWEET_SOURCE_PATH)
  .withColumn("source_file",input_file_name())
  .withColumn("processing_time",current_timestamp())
   )


# COMMAND ----------

#Write_stream 
write_bs = (bronze_stream
            .writeStream
            .option("mergeSchema", "true") 
            .option("checkpointLocation", BRONZE_CHECKPOINT) 
            .start(BRONZE_DELTA) 
  
)


# COMMAND ----------

# Read from the Delta table after the streaming query has started
bronze_df = spark.read.load(BRONZE_DELTA)


# COMMAND ----------

display(bronze_df)

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.0 Bronze Data Exploratory Data Analysis
# MAGIC - How many tweets are captured in your Bronze Table?
# MAGIC - Are there any columns that contain Nan or Null values?  If so how many and what will you do in your silver transforms to address this?
# MAGIC - Count the number of tweets by each unique user handle and sort the data by descending count.
# MAGIC - How many tweets have at least one mention (@) how many tweet have no mentions (@)
# MAGIC - Plot a bar chart that shows the top 20 tweeters (users)
# MAGIC

# COMMAND ----------

# ENTER YOUR CODE HERE
non_null_tweet_count = bronze_df.filter(col("text").isNotNull()).count() #Calculating the non-null tweets values.
print("Number of non-null tweets captured in the Bronze Table:", non_null_tweet_count)

#No Null  tweets .

# COMMAND ----------

null_counts = {col: bronze_df.filter(bronze_df[col].isNull()).count() for col in bronze_df.columns}
print("Null counts in each column:", null_counts)
#No null values in the dataset.

# COMMAND ----------

user_tweet_counts = bronze_df.groupBy("user").agg(count("user").alias("tweet_count"))
user_tweet_counts_sorted = user_tweet_counts.orderBy(desc("tweet_count"))
display(user_tweet_counts_sorted)
# Getting unique users

# COMMAND ----------

has_mention = bronze_df.withColumn("has_mention", when(bronze_df["text"].like("%@%"), "Yes").otherwise("No")) # Getting number of mentions and non--mention tweets
tweet_with_mention_count = has_mention.filter(has_mention["has_mention"] == "Yes").count()# Counting the values
tweet_without_mention_count = has_mention.filter(has_mention["has_mention"] == "No").count()

print("Number of tweets with at least one mention (@):", tweet_with_mention_count)
print("Number of tweets with no mentions (@):", tweet_without_mention_count)


# COMMAND ----------

top_20_users = user_tweet_counts_sorted.limit(20).toPandas() #Plotting the top 20 users
plt.bar(top_20_users["user"], top_20_users["tweet_count"])
plt.xlabel("User")
plt.ylabel("Tweet Count")
plt.title("Top 20 Tweeters")
plt.xticks(rotation=90)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Transform the Bronze Data to Silver Data using a stream
# MAGIC - setup a read stream on your bronze delta table
# MAGIC - setup a write stream to append to the silver delta table
# MAGIC - Use the defined SILVER_CHECKPOINT and SILVER_DELTA paths in the includes
# MAGIC - name your bronze to silver stream as silver_stream
# MAGIC - transform the bronze data to the silver data using the data definition at the top of the notebook

# COMMAND ----------

# timestamp - convert date string in the bronze data to a timestamp
# mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# cleaned_text - the bronze text data with the mentions (@username) removed.
# sentiment - the given sentiment that was associated with the text in the bronze table.


# Define Pandas UDF to extract mentions
@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def extract_mentions_udf(text_series):
    return text_series.str.findall("@\w+")

# Define transformation pipeline
silver_stream = (spark.readStream
                .load(BRONZE_DELTA)
                .withColumn("timestamp", to_timestamp(col("date"), "EEE MMM dd HH:mm:ss z yyyy")) 
                .withColumn("cleaned_text", regexp_replace(col("text"), "@[A-Za-z0-9_]+","")) 
                .withColumn("mention", explode(extract_mentions_udf(col("text")))) 
                .select("timestamp", "cleaned_text", "mention", "sentiment")
                   )

                         
                         

# COMMAND ----------

#Writing the Silver Stream
write_ss = (silver_stream.writeStream
            .option("mergeSchema","true")
            .option("checkPointLocation",SILVER_CHECKPOINT)
            .start(SILVER_DELTA)

)


# COMMAND ----------

silver_df = spark.read.load(SILVER_DELTA)

# COMMAND ----------

display(silver_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.0 Transform the Silver Data to Gold Data using a stream
# MAGIC - setup a read stream on your silver delta table
# MAGIC - setup a write stream to append to the gold delta table
# MAGIC - Use the defined GOLD_CHECKPOINT and GOLD_DELTA paths defines in the includes
# MAGIC - name your silver to gold stream as gold_stream
# MAGIC - transform the silver data to the gold data using the data definition at the top of the notebook
# MAGIC - Load the pretrained transformer sentiment classifier from the MODEL_NAME at the production level from the MLflow registry
# MAGIC - Use a spark UDF to parallelize the inference across your silver data

# COMMAND ----------


# Load model from MLflow registry
model_uri = "models:/HF_TWEET_SENTIMENT/Production"
model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------


gold_stream = (spark.readStream.load(SILVER_DELTA)
        .withColumn("predicted_sentiment", model(col("cleaned_text")))
        .withColumn("sentiment_id", when(col("sentiment") == "positive", 1).otherwise(0))
        .withColumn("predicted_sentiment_id", 
                    when(col("predicted_sentiment.label").isin(["NEU", "NEG"]), 0)
                    .when(col("predicted_sentiment.label") == "POS", 1)
                    .otherwise(0))  
        .select(col("timestamp"), col("mention"), col("cleaned_text"), col("sentiment"), 
                col("predicted_sentiment.label").alias("predicted_sentiment"), col("sentiment_id"), 
                col("predicted_sentiment.score").alias("predicted_score"),col('predicted_sentiment_id'))
)


# COMMAND ----------

write_gs = (gold_stream.writeStream
            .option("mergeSchema","true")
            .option("checkPointLocation",GOLD_CHECKPOINT)
            .start(GOLD_DELTA)

)

# COMMAND ----------

# MAGIC %fs ls "/tmp/labuser104917-3007335/gold.delta"
# MAGIC

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/tmp/labuser104917-3007335/gold.delta/_delta_log/'))


# COMMAND ----------

gold_df = spark.read.load(GOLD_DELTA)

# COMMAND ----------

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0 Capture the accuracy metrics from the gold table in MLflow
# MAGIC Store the following in an MLflow experiment run:
# MAGIC - Store the precision, recall, and F1-score as MLflow metrics
# MAGIC - Store an image of the confusion matrix as an MLflow artifact
# MAGIC - Store the mdoel name and the MLflow version that was used as an MLflow parameters
# MAGIC - Store the version of the Delta Table (input-silver) as an MLflow parameter

# COMMAND ----------

# Calculate evaluation metrics
y_true = gold_df.select("sentiment_id").rdd.flatMap(lambda x: x).collect()
y_pred = gold_df.select("predicted_sentiment_id").rdd.flatMap(lambda x: x).collect()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

# Log metrics and artifacts in MLflow
with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)
    
    # Log confusion matrix as artifact
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cividis")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    
    # Save confusion matrix plot as image
    confusion_matrix_image_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_image_path)
    
    # Log confusion matrix image as artifact
    mlflow.log_artifact(confusion_matrix_image_path, "confusion_matrices")
    
    # Log parameters
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("mlflow_version", mlflow.__version__)
    mlflow.log_param("delta_table_version", 1) #since there is only 1 json file in the delta log.
    



# COMMAND ----------

display(precision)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.0 Application Data Processing and Visualization
# MAGIC - How many mentions are there in the gold data total?
# MAGIC - Count the number of neutral, positive and negative tweets for each mention in new columns
# MAGIC - Capture the total for each mention in a new column
# MAGIC - Sort the mention count totals in descending order
# MAGIC - Plot a bar chart of the top 20 mentions with positive sentiment (the people who are in favor)
# MAGIC - Plot a bar chart of the top 20 mentions with negative sentiment (the people who are the vilians)
# MAGIC
# MAGIC You may want to use the "Loop Application" widget to control whether you repeateded display the latest plots while the data comes in from your streams before moving on to the next section and cleaning up your run.
# MAGIC
# MAGIC *note: A mention is a specific twitter user that has been "mentioned" in a tweet with an @user reference.

# COMMAND ----------

# ENTER YOUR CODE HERE

# Count distinct mentions
distinct_mentions_count = gold_df.select(countDistinct("mention")).collect()[0][0]

# Display the count of distinct mentions
print("Count of distinct mentions:", distinct_mentions_count)




# COMMAND ----------

# Counting neutral, positive, and negative tweets for each mention for the "predicted_sentiment"
mention_sentiment_counts = gold_df.groupBy("mention").agg(
    count(when(col("predicted_sentiment") == "NEU", True)).alias("neutral_count"),
    count(when(col("predicted_sentiment") == "POS", True)).alias("positive_count"),
    count(when(col("predicted_sentiment") == "NEG", True)).alias("negative_count")
)

# COMMAND ----------

display(mention_sentiment_counts)

# COMMAND ----------

gold_df = spark.read.load(GOLD_DELTA)

# COMMAND ----------

mention_data = gold_df.groupBy("mention").agg(
    count("*").alias("total_mentions"),
    count(when(col("predicted_sentiment") == "NEU", True)).alias("neutral_count"),
    count(when(col("predicted_sentiment") == "POS", True)).alias("positive_count"),
    count(when(col("predicted_sentiment") == "NEG", True)).alias("negative_count")
)




# COMMAND ----------

mention_data = mention_data.withColumn(
    "total_count", col("neutral_count") + col("positive_count") + col("negative_count")
)

# Sorting the data by total count in descending order
total_mention_data = mention_data.orderBy(col("total_count").desc())

# COMMAND ----------

display(total_mention_data)

# COMMAND ----------

top_positive_mentions = mention_data.orderBy(col('positive_count').desc()).limit(20).toPandas()

# COMMAND ----------

display(top_positive_mentions)

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(top_positive_mentions["mention"], top_positive_mentions["positive_count"], color='red')
plt.xlabel('Mentions')
plt.ylabel('Positive Count')
plt.title('Top 20 Mentions with Positive Sentiment')
plt.xticks(rotation=90)
plt.show()




# COMMAND ----------

# Filter top 20 mentions with negative sentiment
top_negative_mentions = mention_data.orderBy(col('negative_count').desc()).limit(20).toPandas()


# Plot bar chart for top 20 mentions with negative sentiment
plt.figure(figsize=(10, 6))
plt.bar(top_negative_mentions["mention"], top_negative_mentions["negative_count"], color='red')
plt.xlabel('Mentions')
plt.ylabel('Negative Count')
plt.title('Top 20 Mentions with Negative Sentiment')
plt.xticks(rotation=90)
plt.show()


# COMMAND ----------

application_df = gold_df.groupBy("mention").agg(
    min(col("timestamp")).alias("mintimestamp"),
    max(col("timestamp")).alias("maxtimestamp"),
    count(when(col("predicted_sentiment") == "NEU", True)).alias("neutral_count"),
    count(when(col("predicted_sentiment") == "POS", True)).alias("positive_count"),
    count(when(col("predicted_sentiment") == "NEG", True)).alias("negative_count")
)


# COMMAND ----------

display(application_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.0 Clean up and completion of your pipeline
# MAGIC - using the utilities what streams are running? If any.
# MAGIC - Stop all active streams
# MAGIC - print out the elapsed time of your notebook.

# COMMAND ----------

from pyspark.sql import SparkSession
import time

spark = SparkSession.builder \
    .appName("Active Streaming Queries") \
    .getOrCreate()

streaming_query_manager = spark.streams
active_streams = streaming_query_manager.active
for stream in active_streams:
    try:
        stream.stop()
        print("Stopped stream:", stream.id)
    except Exception as e:
        print("Error stopping stream:", stream.id)
        print("Exception:", e)


#three streams(bronze,silver,gold streams) stopped.

# COMMAND ----------

# Get the notebooks ending time note START_TIME was established in the include file when the notebook started.
END_TIME = time.time()

# COMMAND ----------

END_TIME

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.0 How Optimized is your Spark Application (Grad Students Only)
# MAGIC Graduate students (registered for the DSCC-402 section of the course) are required to do this section.  This is a written analysis using the Spark UI (link to screen shots) that support your analysis of your pipelines execution and what is driving its performance.
# MAGIC Recall that Spark Optimization has 5 significant dimensions of considertation:
# MAGIC - Spill: write to executor disk due to lack of memory
# MAGIC - Skew: imbalance in partition size
# MAGIC - Shuffle: network io moving data between executors (wide transforms)
# MAGIC - Storage: inefficiency due to disk storage format (small files, location)
# MAGIC - Serialization: distribution of code segments across the cluster
# MAGIC
# MAGIC Comment on each of the dimentions of performance and how your impelementation is or is not being affected.  Use specific information in the Spark UI to support your description.  
# MAGIC
# MAGIC Note: you can take sreenshots of the Spark UI from your project runs in databricks and then link to those pictures by storing them as a publicly accessible file on your cloud drive (google, one drive, etc.)
# MAGIC
# MAGIC References:
# MAGIC - [Spark UI Reference Reference](https://spark.apache.org/docs/latest/web-ui.html#web-ui)
# MAGIC - [Spark UI Simulator](https://www.databricks.training/spark-ui-simulator/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ENTER YOUR MARKDOWN HERE
# MAGIC
# MAGIC * Spill - No spill happened in all the jobs. All the stages look perfectly fine all the size of the files looks equal.
# MAGIC * Skew - No skew oberserved in the tasks.
# MAGIC * Shuffle - Due to the usage of the adaptive quey optimization and broadcast joins while using widetransformations like groupBy,aggregate no issues occured related to shuffle. All the the stages looks good where we used wide transformations attached the job stage event timeline in the google docs for the groupby() method.
# MAGIC * Storage - Deserialization time can be affected by storage-related issues, such as reading data from disk or retrieving it from cache. If data is stored inefficiently or requires extensive deserialization, it can contribute to increased deserialization time. Since we used delta table it will join the small files and automatically optimize it but still its affecting it some what for some stages proff attached in the gooogle docs.
# MAGIC * Serialization - I used udfs for for creating dataframes and while loading th emodel i used spark_udf.
# MAGIC
# MAGIC https://drive.google.com/drive/folders/1s3DBpibHyU-FL1EJN8zcYFQy8HUY3gNP?usp=drive_link
# MAGIC
# MAGIC