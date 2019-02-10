# import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import  *  # avg, col, concat, count, desc, explode, lit, min, max, split, stddev, udf, isnan, when
#from pyspark.sql.functions import sum as Fsum
#from pyspark.sql.window import Window
from pyspark.sql import Window
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.clustering import KMeans
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir(r"C:\Users\emu\workspace\dsnd-sparkify")

# create a Spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("Sparkify data") \
    .getOrCreate()

sparkify_data = 'data/mini_sparkify_event_data.json'

df = spark.read.json(sparkify_data)
df.persist()

# =============================================================================
# Clean your dataset, checking for invalid or missing data. For example,
# records without userids or sessionids. In this workspace, the filename is
# mini_sparkify_event_data.json.
# =============================================================================
df.take(5)

df.printSchema()

df.describe("length", "ts").show()

df.count()

df.select("page").dropDuplicates().sort("page").show(df.select("page").count(),False)
df.select("gender").dropDuplicates().sort("gender").show()
df.select("auth").dropDuplicates().sort("auth").show()
df.select("level").dropDuplicates().sort("level").show()
df.select("method").dropDuplicates().sort("method").show()
df.select("status").dropDuplicates().sort("status").show()

df.where(df.userId == "").show()
df.select("userID").distinct().count()

# columns with any NaN values, Null, or missing values
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(col(c) == "", c)).alias(c) for c in df.columns]).show()

# drop rows with missing IDs
# df2 = df.na.drop()
df2 = df.filter(df.userId != "")
df2.select("userID").distinct().count()

df2.describe("length", "ts").show()

# =============================================================================
# Exploratory Data Analysis
# When you're working with the full dataset, perform EDA by loading a small
# subset of the data and doing basic manipulations within Spark. In this
# workspace, you are already provided a small subset of data you can explore.
# 
# Define Churn
# Once you've done some preliminary analysis, create a column Churn to use as
# the label for your model. I suggest using the Cancellation Confirmation
# events to define your churn, which happen for both paid and free users. As a
# bonus task, you can also look into the Downgrade events.
# 
# Explore Data
# Once you've defined churn, perform some exploratory data analysis to observe
# the behavior for users who stayed vs users who churned. You can start by
# exploring aggregates on these two groups of users, observing how much of a
# specific action they experienced per a certain time unit or number of songs
# played.
# =============================================================================

# These both work in the same way, but the second breaks the rest of the code
df2 = df2.withColumn("churn", when(df.page == "Cancellation Confirmation", 1).otherwise(0))
 
#cancel_udf = udf(lambda  x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
#df2 = df2.withColumn("churn", cancel_udf(df2.page))
df2.printSchema()
df2.groupBy().sum().collect()[0]

df2.filter(df2.churn == 1).show()
df2.select(["artist","itemInSession","length", "level","page","sessionId","churn","ts"]).filter(df2.userId == 18).sort(asc('ts')).show(1000)

# investigating the Downgrade too
df3 = df.withColumn("churn", when(df.page == "Downgrade", 1).otherwise(0))
df3.filter(df3.churn == 1).show()
df3.select(["artist","itemInSession","length", "level","page","sessionId","churn","ts"]).filter(df3.userId == 95).sort(asc('ts')).show(1000)


# =============================================================================
# Feature ideas:
#     time from first event until cancelation
#     number of songs played
#     number of artists
#     number of friends added
#     number of sessions
#     number of errors
#     ratio of thumbs up to down
#     number of songs in playlist
#     gender dummy
#     browser dummy
# =============================================================================
    

#unction = udf(lambda canceled : int(canceled == "Cancellation Confirmation"), IntegerType())

user_window = Window \
    .partitionBy('userID') \
    .orderBy(desc('ts')) \
    .rangeBetween(Window.unboundedPreceding, 0)

cusum = df.filter((df.page == 'NextSong') | (df.page == 'Cancellation Confirmation')) \
    .select('userID', 'page', 'ts') \
    .withColumn('cancelvisit', when(df.page == "Cancellation Confirmation", 1).otherwise(0)) \
    .withColumn('period', Fsum('cancelvisit').over(user_window))

cusum.filter((cusum.page == 'NextSong')) \
    .groupBy('userID', 'period') \
    .agg({'period': 'count'}) \
    .agg({'count(period)': 'avg'}).show()

#get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour, IntegerType())
#song_time = df2.withColumn("hour", get_hour(df2.ts))
song_time = df2.withColumn("hour", hour(from_unixtime(col('ts')/1000)))

songs_in_hour = song_time.filter(song_time.page == "NextSong") \
                .groupby(song_time.hour).count() \
                .orderBy(song_time.hour.cast("float"))

songs_in_hour.take(5)

songs_in_hour_pd = songs_in_hour.toPandas()
songs_in_hour_pd.hour = pd.to_numeric(songs_in_hour_pd.hour)

plt.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])
plt.xlim(-1, 24);
plt.ylim(0.99 * np.min(songs_in_hour_pd["count"]), 1.05 * np.max(songs_in_hour_pd["count"]))
plt.xlabel("Hour")
plt.ylabel("Songs played");

#test = df.select('ts', hour(from_unixtime(col('ts')/1000)).alias('hour'))
#test = df.select('ts', from_unixtime(col('ts')/1000).alias('datetime'))
#test.printSchema()
#test.select(hour('datetime').alias('hour')).show()

# Making the churn feature display 1 for all rows of a user who canceled

df_churn = df2.withColumn("churn", sum("churn").over(user_window))
#df_churn.select(["userId", "firstname", "ts", "page", "level", "churn"]).where(df_churn.userId == "18").sort("ts").collect()
#df_churn.select(["userId", "page", "churn"]).where(df_churn.userId == "18").sort("ts").show(1000)

### Adding the hour when the song was played
df_churn = df_churn.withColumn("hour", hour(from_unixtime(col('ts')/1000)))

### Gender dummy
df_churn.cube(["gender"]).count().show()
#df_churn = df_churn.withColumn("gender_dum", when(df.gender == "M", 1).otherwise(when(df.gender == "F", 0).otherwise("null")))

indexer = StringIndexer(inputCol="gender", outputCol="gender_dum")
df_churn = indexer.fit(df_churn).transform(df_churn)
#indexed.select(["userId","gender","genderIndex"]).groupby("userId").max().show()
df_churn.cube(["gender_dum"]).count().show()

### Browser dummy
df_churn.cube(["userAgent"]).count().show(df_churn.select("userAgent").count(),False)
# find user access agents, and perform one-hot encoding on the user 
userAgents = df_churn.select(['userId', 'userAgent']).distinct()
userAgents = userAgents.fillna('N/A')
# build string indexer
strIndexer = StringIndexer(inputCol="userAgent", outputCol="userAgentInd")
model = strIndexer.fit(df_churn)
df_churn = model.transform(df_churn)
# one hot encode userAgent column
encoder = OneHotEncoder(inputCol="userAgentInd", outputCol="userAgentVec")
df_churn = encoder.transform(df_churn).drop(*["userAgentInd", "userAgent"])

### Days of activity

# Find minimum/maximum time stamp of each user
min_timestmp = df_churn.select(["userId", "ts"]).groupby("userId").min("ts")
max_timestmp = df_churn.select(["userId", "ts"]).groupby("userId").max("ts")

# Find days active of each user
daysActive = min_timestmp.join(max_timestmp, on="userId")
daysActive = daysActive.withColumn("days_active", 
                                   datediff(from_unixtime(col('max(ts)')/1000),
                                            from_unixtime(col('min(ts)')/1000))).select(["userId", "days_active"]) 
df_churn = df_churn.join(daysActive, on = 'userId', how = 'left')

### Number of sessions
n_sessions = df_churn.select(["userId", "sessionId"]).distinct().groupby("userId").count().sort(["userId", "count"])
n_sessions = n_sessions.withColumnRenamed("count", "n_sessions")

df_churn = df_churn.join(n_sessions, on = 'userId', how = 'left')

### Average numer of songs per session

avg_session_songs = df_churn.filter((df_churn.page == 'NextSong')) \
                    .groupBy('userID', 'sessionId') \
                    .agg({'song': 'count'}) \
                    .groupBy('userID') \
                    .agg({'count(song)': 'avg'})
avg_session_songs = avg_session_songs.withColumnRenamed("avg(count(song))", "avg_sess_songs")

df_churn = df_churn.join(avg_session_songs, on = 'userId', how = 'left')

# Renaming churn to label
df_churn = df_churn.withColumnRenamed("churn", "label")

feature_names = ["hour", "gender_dum", "userAgentVec", "days_active", "n_sessions", "avg_sess_songs"] # df_churn.schema.names

# =============================================================================
# Modeling
# Split the full dataset into train, test, and validation sets. Test out
# several of the machine learning methods you learned. Evaluate the accuracy
# of the various models, tuning parameters as necessary. Determine your winning
# model based on test accuracy and report results on the validation set. Since
# the churned users are a fairly small subset, I suggest using F1 score as the
# metric to optimize.
# =============================================================================

# =============================================================================
# Build pipeline
# =============================================================================

train, test = df_churn.randomSplit([0.2, 0.8], 42)
lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)
assembler = VectorAssembler(inputCols=feature_names, outputCol="features_vec")
# Scale each column
scalar = MinMaxScaler(inputCol="features_vec", outputCol="features")
#df = assembler.transform(df)
pipeline = Pipeline(stages=[assembler, scalar, lr])

#model = pipeline.fit(train)

#results = model.transform(test)

#paramGrid = ParamGridBuilder() \
#    .addGrid(lr.regParam, [0.0, 0.1]) \
#    .addGrid(lr.maxIter, [10]) \
#    .build()
#
#crossval = CrossValidator(estimator=pipeline,
#                          estimatorParamMaps=paramGrid,
#                          evaluator=MulticlassClassificationEvaluator(metricName="f1"),
#                          numFolds=3)
#
#lrModel = crossval.fit(train)

accs = lrModel.avgMetrics
best_params = lrModel.bestModel.stages[-1].extractParamMap()

results = lrModel.transform(test)

print(results.filter(results.label == results.prediction).count())
print(results.count())
print("The highest accuracy is {:2.2%}".format(results.filter(results.label == results.prediction).count() / results.count()))


def modelEvaluator(model, metric, data):
    """
        Input: 
            model - pipeline of a list of transformers
            metric - the metric of the evaluation (f1, accuracy, etc..)
            data - dataset used in the evaluation
        Output:
            list of [score, confusion matrix]
        Description:
            Evaluate a model's performance with our metric of choice
    """
    # generate predictions
    evaluator = MulticlassClassificationEvaluator(metricName = metric)
    predictions = model.transform(data)
    
    # calcualte score
    score = evaluator.evaluate(predictions)
    confusion_matrix = (predictions.groupby("label")
                                   .pivot("prediction")
                                   .count()
                                   .toPandas())
    return [score, confusion_matrix]

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 0.1]) \
    .addGrid(lr.maxIter, [10]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                          numFolds=3)

lrModel = crossval.fit(train)

f1_lr, conf_mtx_lr = modelEvaluator(lrModel, 'f1', test)
print('The F1 score for the logistic regression model:', f1_lr)
conf_mtx_lr

###
rf = RandomForestClassifier(numTrees = 50,  featureSubsetStrategy='auto')
pipeline = Pipeline(stages=[assembler, scalar, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxBins, [16, 32]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                          numFolds=3)

rfModel = crossval.fit(train)

f1_rf, conf_mtx_rf = modelEvaluator(rfModel, 'f1', test)
print('The F1 score for the random forest model:', f1_rf)
conf_mtx_rf

###
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
pipeline = Pipeline(stages=[assembler, scalar, nb])

paramGrid = ParamGridBuilder() \
    .addGrid(nb.regParam , [0.1, 0.2]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                          numFolds=3)

nbModel = crossval.fit(train)

f1_nb, conf_mtx_nb = modelEvaluator(nbModel, 'f1', test)
print('The F1 score for the naive bayes model:', f1_nb)
conf_mtx_nb



# =============================================================================
# Final Steps
# Clean up your code, adding comments and renaming variables to make the code
# easier to read and maintain. Refer to the Spark Project Overview page and
# Data Scientist Capstone Project Rubric to make sure you are including all
# components of the capstone project and meet all expectations. Remember, this
# includes thorough documentation in a README file in a Github repository, as
# well as a web app or blog post.
# =============================================================================
