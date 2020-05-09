import json
from pyspark import SparkContext, SparkConf

sc = SparkContext("local","PySpark Word Count Exmaple")

words = sc.textFile("/home/ubuntu/data/trump_tweets.txt").map(lambda line: json.loads(line)["user"]["name"])
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b).sortBy(lambda (word, count): count)
wordCounts.saveAsTextFile("/home/ubuntu/pyAcda/outputTweetsPerUser/")

