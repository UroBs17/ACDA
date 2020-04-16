import json
from pyspark import SparkContext, SparkConf
l=[ "TRUMP", "DICTATOR", "MAGA", "IMPEACH", "DRAIN","SWAP", "CHANGE"]
sc = SparkContext("local","PySpark Word Count Exmaple")
words = sc.textFile("/home/ubuntu/data/trump_tweets.txt").flatMap(lambda line: json.loads(line)["text"].upper().split(" ")).filter(lambda word: word in l)
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b).sortBy(lambda (word, count): count)

wordCounts.saveAsTextFile("/home/ubuntu/pyAcda/outputSpecificWords/")

