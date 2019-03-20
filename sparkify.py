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
plt.rcParams['figure.figsize'] = 12, 9

import os
os.chdir(r"C:\Users\tarek\git_workspace\dsnd-sparkify")

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
df.where(df.userId == "").select("page").distinct().show()
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
df2.select(["artist","itemInSession","length", "level","page","sessionId","churn","ts"]).filter(df2.userId == 18).sort(asc('ts')).show()

# investigating the Downgrade too
df3 = df.withColumn("churn", when(df.page == "Downgrade", 1).otherwise(0))
df3.filter(df3.churn == 1).show()
df3.select(["artist","itemInSession","length", "level","page","sessionId","churn","ts"]).filter(df3.userId == 95).sort(asc('ts')).show()


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
    

# Initializing final table

df_user = df2.select(["userId"]).distinct().sort(["userId"])

churned = df2.filter(df2.churn == 1).select('userId').distinct().sort(["userId"])

churned_array = [int(row.userId) for row in churned.collect()]

# add a new churned column to the final table
df_user = df_user.withColumn('churn', when(df_user.userId.isin(*churned_array), 1).otherwise(0))

# Calculate the number of songs played per hour of the day
user_window = Window \
    .partitionBy('userID') \
    .orderBy(desc('ts')) \
    .rangeBetween(Window.unboundedPreceding, 0)

cusum = df.filter((df.page == 'NextSong') | (df.page == 'Cancellation Confirmation')) \
    .select('userID', 'page', 'ts') \
    .withColumn('cancelvisit', when(df.page == "Cancellation Confirmation", 1).otherwise(0)) \
    .withColumn('period', sum('cancelvisit').over(user_window))

cusum.filter((cusum.page == 'NextSong')) \
    .groupBy('userID', 'period') \
    .agg({'period': 'count'}) \
    .agg({'count(period)': 'avg'}).show()

#get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour, IntegerType())
#song_time = df2.withColumn("hour", get_hour(df2.ts))
song_time = df2.filter(df2.page == "NextSong").withColumn("hour", hour(from_unixtime(col('ts')/1000)))

songs_in_hour = song_time.filter(song_time.page == "NextSong") \
                .groupby(song_time.hour).count() \
                .orderBy(song_time.hour.cast("float"))

songs_in_hour.take(5)

songs_in_hour_pd = songs_in_hour.toPandas()
songs_in_hour_pd.hour = pd.to_numeric(songs_in_hour_pd.hour)


with plt.xkcd():

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.annotate(
        'THE HEIGHT OF \nMUSIC PLAYBACK\nIS AT HIGH NOON',
        xy=(12, 11530), arrowprops=dict(arrowstyle='->'), xytext=(11, 9500))

    plt.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])

    plt.xlabel('hour')
    plt.ylabel('songs played')
    fig.text(
        0.5, 0.05,
        'Total number of songs played throughout the day',
        ha='center')

#plt.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])
#plt.xlim(-1, 24);
#plt.ylim(0.99 * np.min(songs_in_hour_pd["count"]), 1.05 * np.max(songs_in_hour_pd["count"]))
#plt.xlabel("Hour")
#plt.ylabel("Songs played")
#plt.title('Distribution of number of songs played throughout the day');

#test = df.select('ts', hour(from_unixtime(col('ts')/1000)).alias('hour'))
#test = df.select('ts', from_unixtime(col('ts')/1000).alias('datetime'))
#test.printSchema()
#test.select(hour('datetime').alias('hour')).show()

# Making the churn feature display 1 for all rows of a user who canceled

#df_churn = df2.withColumn("churn", sum("churn").over(user_window))
#df_churn.select(["userId", "firstname", "ts", "page", "level", "churn"]).where(df_churn.userId == "18").sort("ts").collect()
#df_churn.select(["userId", "page", "churn"]).where(df_churn.userId == "18").sort("ts").show(1000)

### Adding the hour when the song was played
#df_user = df_churn.withColumn("hour", hour(from_unixtime(col('ts')/1000)))

avg_hour_of_play = song_time.select(["userId", "hour"]) \
                    .groupBy('userID') \
                    .agg({'hour': 'avg'})

avg_hour_of_play = avg_hour_of_play.withColumnRenamed("avg(hour)", "avg_play_hour")

df_user = df_user.join(avg_hour_of_play, on = 'userId', how = 'left')

(df_user.groupby("churn").avg().select('churn', 'avg(avg_play_hour)')
                .toPandas()
                .set_index('churn')
                .plot(kind = 'bar', title = 'Average hour of play churned/Non-churned users'))

### Gender dummy (1: male, 0: female)
df2.cube(["gender"]).count().show()
#df_churn = df_churn.withColumn("gender_dum", when(df.gender == "M", 1).otherwise(when(df.gender == "F", 0).otherwise("null")))

indexer = StringIndexer(inputCol="gender", outputCol="gender_dum")
df2 = indexer.fit(df2).transform(df2)
#indexed.select(["userId","gender","genderIndex"]).groupby("userId").max().show()
df2.cube(["gender_dum"]).count().show()

gender_dum = df2.select(["userId", "gender_dum"]).distinct().sort("userId")

df_user = df_user.join(gender_dum, on = 'userId', how = 'left')

#(df_user.groupby(["churn", "gender_dum"]).count()
#                .toPandas()
#                .set_index('churn')
#                .plot(kind = 'bar', title = 'Average hour of play Churned/Non-Churned users'))

df_user.groupby(["churn"]).avg().show()


### Browser dummy
#df2.cube(["userAgent"]).count().show(df2.select("userAgent").count(),False)
#df2.groupby(["userAgent"]).count().sort(desc("count")).toPandas()
# find user access agents, and perform one-hot encoding on the user 
#userAgents = df2.select(['userId', 'userAgent']).distinct()
#userAgents = userAgents.fillna('N/A')
## build string indexer
#strIndexer = StringIndexer(inputCol="userAgent", outputCol="userAgentInd")
#model = strIndexer.fit(df2)
#df2 = model.transform(df2)
## one hot encode userAgent column
#encoder = OneHotEncoder(inputCol="userAgentInd", outputCol="userAgentVec")
#user_agent = encoder.transform(df2).drop(*["userAgentInd", "userAgent"]).select(["userId", "userAgentVec"]).distinct()

### Days of activity

# Find minimum/maximum time stamp of each user
min_timestmp = df2.select(["userId", "ts"]).groupby("userId").min("ts")
max_timestmp = df2.select(["userId", "ts"]).groupby("userId").max("ts")

# Find days active of each user
daysActive = min_timestmp.join(max_timestmp, on="userId")
daysActive = daysActive.withColumn("days_active", 
                                   datediff(from_unixtime(col('max(ts)')/1000),
                                            from_unixtime(col('min(ts)')/1000))).select(["userId", "days_active"]) 
df_user = df_user.join(daysActive, on = 'userId', how = 'left')

(df_user.groupby(["churn"]).avg().select(["churn", "avg(days_active)"])
                .toPandas()
                .set_index('churn')
                .plot(kind = 'bar', title = 'Average days active for churned/Non-churned users'))

### Number of sessions
n_sessions = df2.select(["userId", "sessionId"]).distinct().groupby("userId").count().sort(["userId", "count"])
n_sessions = n_sessions.withColumnRenamed("count", "n_sessions")

df_user = df_user.join(n_sessions, on = 'userId', how = 'left')

(df_user.groupby(["churn"]).avg().select(["churn", "avg(n_sessions)"])
                .toPandas()
                .set_index('churn')
                .plot(kind = 'bar', title = 'Average number of sessions for churned/Non-churned users'))

### Average number of songs per session

avg_session_songs = df2.filter((df2.page == 'NextSong')) \
                    .groupBy('userID', 'sessionId') \
                    .agg({'song': 'count'}) \
                    .groupBy('userID') \
                    .agg({'count(song)': 'avg'})
#df_churn1 = df_churn.groupby(["userId"]).max("ts").show()

#df_churn1 = df_churn.as[Record]
#  .groupByKey(_.userId)
#  .reduceGroups((x, y) => if (x.ts > y.ts) x else y)
#df_churn1 = df_churn.groupByKey(_.userId).reduceGroups((x, y) => if (x.ts > y.ts) x else y)

avg_session_songs = avg_session_songs.withColumnRenamed("avg(count(song))", "avg_sess_songs")

df_user = df_user.join(avg_session_songs, on = 'userId', how = 'left')

(df_user.groupby(["churn"]).avg().select(["churn", "avg(avg_sess_songs)"])
                .toPandas()
                .set_index('churn')
                .plot(kind = 'bar', title = 'Average number of songs per session for churned/Non-churned users'))

### Number of errors

errors = df2.select(["userId", "page"]).filter((df2.page == 'Error')).groupby("userId").agg({'page': 'count'})
errors = errors.withColumnRenamed("count(page)", "n_errors")

df_user = df_user.join(errors, on = 'userId', how = 'left')

(df_user.groupby(["churn"]).avg().select(["churn", "avg(n_errors)"])
                .toPandas()
                .set_index('churn')
                .plot(kind = 'bar', title = 'Average number of errors for churned/Non-churned users'))

# Renaming churn to label
df_user = df_user.withColumnRenamed("churn", "label")

# setting missing values to zero
df_user = df_user.fillna(0)

# =============================================================================
# Modeling
# Split the full dataset into train, test, and validation sets. Test out
# several of the machine learning methods you learned. Evaluate the accuracy
# of the various models, tuning parameters as necessary. Determine your winning
# model based on test and report results on the validation set. Since
# the churned users are a fairly small subset, I suggest using F1 score as the
# metric to optimize.
# =============================================================================

# =============================================================================
# Build pipeline
# =============================================================================

train, test = df_user.randomSplit([0.8, 0.2], 42)
#lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)

feature_names = ["avg_play_hour", "gender_dum", "days_active", "n_sessions", "avg_sess_songs", "n_errors"] # df_churn.schema.names

# Vectorize the features
assembler = VectorAssembler(inputCols=feature_names, outputCol="features_vec")
# Scale each column
scalar = MinMaxScaler(inputCol="features_vec", outputCol="features")
#df = assembler.transform(df)
#pipeline = Pipeline(stages=[assembler, scalar, lr])

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

#accs = lrModel.avgMetrics
#best_params = lrModel.bestModel.stages[-1].extractParamMap()
#
#results = lrModel.transform(test)
#
#print(results.filter(results.label == results.prediction).count())
#print(results.count())
#print("The highest accuracy is {:2.2%}".format(results.filter(results.label == results.prediction).count() / results.count()))


def model_evaluator(model, metric, data):
    """
        args: 
            model - pipeline of a list of transformers
            metric - the metric of the evaluation (f1, accuracy, etc..)
            data - dataset used in the evaluation
        returns:
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
                                   .count())
    return [score, confusion_matrix]


def pipe_run_f1(model, param_grid, train_data, test_data):
    """
        args: 
            model - pipeline of a list of transformers
            param_grid - a grid of hyperparameters to be tested out
            data - dataset used in the evaluation
        returns:
            trained model
        Description:
            Trains the selected model on our data
    """
    pipeline = Pipeline(stages=[assembler, scalar, model])
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=param_grid,
                              evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                              numFolds=3)
    out_model = crossval.fit(train_data)
    
    evaluator = MulticlassClassificationEvaluator(metricName = "f1")
    predictions = out_model.transform(test_data)
    # calcualte score
    score = evaluator.evaluate(predictions)
    confusion_matrix = (predictions.groupby("label")
                                   .pivot("prediction")
                                   .count())

    return out_model, score, confusion_matrix


lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)

param_grid = ParamGridBuilder().build()
#    .addGrid(lr.regParam, [0.0, 0.1]) \
#    .addGrid(lr.maxIter, [10]) \
#    .build()

lr_model, score, conf_mtx_lr = pipe_run_f1(lr, param_grid, train, test)  # crossval.fit(train)

#pipeline = Pipeline(stages=[assembler, scalar, lr])
#lr_model = pipeline.fit(train)
#evaluator = MulticlassClassificationEvaluator(metricName = "f1")
#predictions = lr_model.transform(test)
## calculate score
#score = evaluator.evaluate(predictions)
#confusion_matrix = (predictions.groupby("label")
#                               .pivot("prediction")
#                               .count()
#                               .show())

 
f1_lr, conf_mtx_lr = model_evaluator(lr_model, 'f1', test)
print('The F1 score for the logistic regression model:', f1_lr)
conf_mtx_lr.show()

###
rf = RandomForestClassifier(numTrees = 50, featureSubsetStrategy='auto')

param_grid = ParamGridBuilder().build()
#    .addGrid(rf.maxBins, [16, 32]) \
#    .build()

rf_model = pipe_run_f1(rf, param_grid, train)

f1_rf, conf_mtx_rf = model_evaluator(rf_model, 'f1', test)
print('The F1 score for the random forest model:', f1_rf)
conf_mtx_rf

###
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

param_grid = ParamGridBuilder().build()
    #.addGrid(nb.regParam , [0.1, 0.2]) \
    #.build()

nb_model = pipe_run_f1(nb, param_grid, train)

f1_nb, conf_mtx_nb = model_evaluator(nb_model, 'f1', test)
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
