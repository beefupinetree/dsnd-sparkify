# Capstone Spark Project - Sparkify

### Motivation

As with any online streaming service, user activities on the site are logged. This creates huge datasets needing to be cleaned and analyzed in order to extract useful information. As powerful as present-day PCs have become, they are yet to be powerful enough to handle such massive sets of data. A standalone PC would not have enough RAM or CPU power to process the data within a reasonable timeframe, which is where Spark comes in. Spark is a framework used to create a cluster composed of multiple PCs. The data is then partitioned and distributed throughout the cluster to be processed in parallel.

### Objective

We will use Spark to analyze the churning behavior of the users of Sparkify. We will attempt to predict the likelihood of a user cancelling their service by using different machine learning models. We will compare the accuracy of each given model and then pick the most accurate and run it on the full 12GB dataset on a spark cluster on Amazon Web Services.

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

### Dependencies

* pyspark
* Numpy
* Pandas
* matplot

### File description

* sparkify_local.ipynb / .html: The Jupyter notebook and HTML file containing the code used on the sample dataset which was tested on a standalone PC
* sparkify_aws.ipynb / .html: The Jupyter notebook and HTML file with the code applied on the full dataset using a cluster of 6 machines on AWS

### Summary results

The Random Forest Classifier was the best performing model out the four we chose. With our 6 engineered features, we were able to obtain an F1 score of 0.88 on the 'small' dataset when we ran it on a local machine. Unfortunately, running the same code on the 12GB of data on an AWS cluster consistently produced errors and did not yielf any results.