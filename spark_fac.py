from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS

sparkC = SparkContext()
sqlC = SQLContext(sparkC)
trainData = sparkC.textFile("Data/trainData.txt").map(lambda line: line.split("\t"))
testData = sparkC.textFile("Data/testData.txt").map(lambda line: line.split("\t"))

trainDataFrame = sqlC.createDataFrame(trainData,["user","item","rating"])
testDataFrame = sqlC.createDataFrame(testData,["user","item"])

als = ALS(rank = 10,maxIter = 5)

model = als.fit(trainDataFram)

predTestData = model.transform(testDataFrame)

prediction = sorted(predTestData.collect(), key = lambda r: int(r[0]))

with open("Result/prediction1.txt","w") as predFile:
    for line in prediction:
        if line[2]!=line[2]:
            temp_str = "None"
        else:
            temp_str = str(int(line[2]))
        predFile.write(str(line[0])+"|"+str(line[1])+"|"+temp_str+"\n")

sparkC.stop()
