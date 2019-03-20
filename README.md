# Capstone Spark Project - Sparkify

### Motivation

As with any online streaming service, user activities on the site are logged which creates huge datasets to be cleaned and analyzed in order to get useful information out of it. As powerful as present-day PCs have become, they are yet to be powerful enough to handle such massive sets of data. A standalone PC would not have enough RAM or CPU power to process the data within a reasonable timeframe, which is where Spark comes in. Spark is a framework used to create a cluster composed of multiple PCs. The data is then partitioned and distributed throughout the cluster to be processed in parallel.

### Objective

We will use Spark to analyze the churning behavior of the users of Sparkify. We will try to predict the likelihood of a user canceling their service by using several different machine learning models. We will compare the accuracy of each given model and then pick the most accurate and run it on the full 12GB dataset on a spark cluster on Amazon Web Services.

### The Data

The data at our disposal is a very granular log of the activities on the service's interface. It contains demographic information, user selection, timestamps, and more. Some of the available features are:

* Artist
* Authorization
* FirstName
* Gender
* Lastname
* Song length
* Payment level
* Location
* page
* Song
* Timestamp

### Packages used

* pyspark
* Numpy
* Pandas
* matplot

### File description

* sparkify_local.ipynb: The Jupyter notebook containing the code used on the sample dataset
* sparkify_aws.ipynb

### Summary results

The presented model represents the best model I have constructed so far. Originally I only used the all the activities in *page* as features, which yielded 0.69 F1 score on the small test set we have. After I engineered 6 other features as noted in the project, I was able to obtain an F1 score of 0.80 (0.88 after scale up to the large dataset).
