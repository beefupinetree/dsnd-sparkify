# import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import  *  # avg, col, concat, count, desc, explode, lit, min, max, split, stddev, udf, isnan, when
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DateType
from pyspark.ml.feature import RegexTokenizer, VectorAssembler, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

# columns with any NaN values, Null, or missing values
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()
df.select([count(when(col(c) == "", c)).alias(c) for c in df.columns]).show()

# drop rows with missing IDs
# df2 = df.na.drop()
df2 = df.filter(df.userId != "")

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


get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour, IntegerType())
song_time = df2.withColumn("hour", get_hour(df2.ts))

songs_in_hour = song_time.filter(song_time.page == "NextSong") \
                .groupby(song_time.hour).count() \
                .orderBy(song_time.hour.cast("float"))

songs_in_hour.take(5)

songs_in_hour_pd = songs_in_hour.toPandas()
songs_in_hour_pd.hour = pd.to_numeric(songs_in_hour_pd.hour)

plt.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])
plt.xlim(-1, 24);
plt.ylim(0, 1.2 * max(songs_in_hour_pd["count"]))
plt.xlabel("Hour")
plt.ylabel("Songs played");

test = df.select('ts', hour(from_unixtime(col('ts')/1000)).alias('hour'))
test = df.select('ts', from_unixtime(col('ts')/1000).alias('datetime'))
test.printSchema()
test.select(hour('datetime').alias('hour')).show()
