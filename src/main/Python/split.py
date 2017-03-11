#! /usr/bin/env python

import sys
import jieba
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql import Row
import numpy as np
import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import os
from pyspark.sql.types import *
from py4j.protocol import Py4JJavaError
import codecs
import shutil
from pyspark.sql import functions as F
from pyspark.mllib.linalg import Vectors
import pyspark.mllib
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vector as MLVector, Vectors as MLVectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.ml import linalg as ml_linalg
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from numpy import array
from math import sqrt


# Get for all the patients folders. 
root = "/Users/yanli/Downloads/big_data/data2/"
dirs = next( os.walk(root) )[1]

#conf = SparkConf().setAppName("impression")
#sc = SparkContext('local[4]', '', conf=conf)
#spark = SparkSession.builder.config(conf = conf).appName("spark session").getOrCreate()

sqlContext = SQLContext(sc)
patients_index= spark.read.csv(root + "PATIENT_INDEX.csv", header=True)

# find empty files
def try_load(path):
    rdd = sc.textFile(path)
    try:
        rdd.first()
        return rdd
    except ValueError as e:
        return sc.emptyRDD()

# split words from sentences in exam results. 
def sep_words(line):
	words_list = [ ]
	for phrase in line:
		try:
			words = jieba.cut(phrase, cut_all=False)
			for word in words:
				if len(word) > 1:
					words_list.append(word)				
		except AttributeError:
			j = 1	
			print("AttributeError:", i)	
	return '/ '.join(words_list)

# Initial value for words and label. 
i = 0	
labels = []
all_words = []

# Append results in every sub-directory
for dir in dirs:
	rdd = try_load(root + dir + "/" + "EXAM_RESULT.CSV")
	i += 1
	print(i)
	if not rdd.isEmpty():
		j = 0
		df = spark.read.csv(root + dir + "/" + "EXAM_RESULT.CSV", header=True)
		impression = df.select("IMPRESSION")
		imp = impression.rdd	
		words = imp.map(lambda l : sep_words(l))
		if os.path.exists(root + dir + "/" + "impression"):
			try: 
				shutil.rmtree(root + dir + "/" + "impression", ignore_errors=True)
			except OSError:
				pass			
		words.saveAsTextFile(root + dir + "/" + "impression")
		words2 = open(root + dir + "/" + "impression" + "/part-00000", 'r').read()
		words3 = ' '.join(' '.join(words2.split('\n')).split('/')).split()
		words4 = ""
		for word in words3:
			words4 = words4 + " " + word
		all_words.append(words4)
		print("all words: ", len(all_words))
		if j == 0:
			label = patients_index.select("TREAT_RESULT", "PATIENT_ID").rdd.map(lambda x : (x[0], x[1])).collect()[i]
			labels.append(label)
			print("labels: ", len(labels))	        	        	

# Create dataset with combined information. 
results = sc.parallelize(labels)
results2 = results.map(lambda x : x[0])
patient_id = results.map(lambda x : x[1])
impression = sc.parallelize(all_words) 
impression_rdd = impression.zip(results2)         
colname = ["impression", "lables"]
df.imp = spark.createDataFrame(impression_rdd, colname)
df.imp.write.csv('/Users/yanli/Downloads/big_data/data2_impression')
colname2 = ["label", "patient_id"]
patient_id = spark.createDataFrame(results, colname2)
patient_id.write.csv('/Users/yanli/Downloads/big_data/data2_patient_id')

# Convert text lable to 0 or 1. 
df.imp2 = df.imp.withColumn("lables", F.when(df.imp["lables"] == "治愈", 1).otherwise(0))
# Tokenize words.
tokenizer = Tokenizer(inputCol="impression", outputCol="words")
wordsData = tokenizer.transform(df.imp2)
# Calculate TF.
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2048)
featurizedData = hashingTF.transform(wordsData)
# CalculateTF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# prepare data for Naive Bayes model.
def as_mllib(v):
    if isinstance(v, ml_linalg.SparseVector):
        return MLLibVectors.sparse(v.size, v.indices, v.values)
    elif isinstance(v, ml_linalg.DenseVector):
        return MLLibVectors.dense(v.toArray())
    else:
        raise TypeError("Unsupported type: {0}".format(type(v)))

trainDataRdd = rescaledData.select("lables", "features").rdd.map(lambda line : LabeledPoint(line[0],as_mllib(line[1]))) 
#trainDataRdd.saveAsTextFile('/Users/yanli/Downloads/big_data/rescaled_impression')

# Oversample the data. 
trainDataRdd2 = trainDataRdd.union(trainDataRdd.filter(lambda x : x.label == 0.0)).union(trainDataRdd.filter(lambda x : x.label == 0.0)).union(trainDataRdd.filter(lambda x : x.label == 0.0)).union(trainDataRdd.filter(lambda x : x.label == 0.0)).union(trainDataRdd.filter(lambda x : x.label == 0.0))
training, test = trainDataRdd2.randomSplit([0.6, 0.4], seed = 0)

# train a naive bayes model.
NBmodel = NaiveBayes.train(training, 1.0)

# Make prediction and evaluate the model.
predictionAndLabel = test.map(lambda p : (NBmodel.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x: 1.0 if x[0] == x[1] else 0.0).count() / test.count()
print('model accuracy {}'.format(accuracy))
metrics = BinaryClassificationMetrics(predictionAndLabel)
# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)
# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)

# Save the model
output_dir = '/Users/yanli/Downloads/big_data/ImpressionNaiveBayesModel'
shutil.rmtree(output_dir, ignore_errors=True)
NBmodel.save(sc, output_dir)


# To predict a new entry.
data = spark.read.csv(root + dirs[4] + "/" + "EXAM_RESULT.CSV", header=True)
impression = data.select("IMPRESSION")
imp = impression.rdd	
words = imp.map(lambda l : sep_words(l))
words.saveAsTextFile(root + "/" + "impression")
words2 = words.map(lambda x : ' '.join(x.split('/'))).collect()
words3 = ""
for word in words2:
	words3 = words3 + " " + word
new_words = []
new_words.append(words3)
new_wordsRdd = sc.parallelize(new_words).map(lambda l : Row(l))
column = ["impression"]
wordDF = spark.createDataFrame(new_wordsRdd, column)
tokenizer = Tokenizer(inputCol="impression", outputCol="words")
newwordsData = tokenizer.transform(wordDF)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=256)
featurizedData = hashingTF.transform(newwordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
newData = idfModel.transform(featurizedData)
sameModel = NaiveBayesModel.load(sc, output_dir)
newInput = newData.select("features").rdd.map(lambda line: as_mllib(line[0]))
print('NaiveBayes Model Predict:',sameModel.predict(newInput).take(1))
# Read and parse EMR data.
#emr = sc.textFile('/Users/yanli/Downloads/big_data/data2/0292761/0292761-1-6.txt', use_unicode=False).map(lambda x : x.decode('utf-16le'))
#t = codecs.open('/Users/yanli/Downloads/big_data/data2/0292761/0292761-1-6.txt', encoding='utf-16le', errors='replace').read()

# For Kmeans clustering
parsedData = trainDataRdd2.map(lambda l : array(l.features))
clusters = KMeans.train(parsedData, 4, maxIterations=10, initializationMode="random")
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)




