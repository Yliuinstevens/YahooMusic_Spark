# Import libraries
from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
import time
import numpy as np

##################################################################
# Functions
def read_lines(file, num):
	lines = []
	line = file.readline()
	if line:
		lines.append(line.strip().split("|"))
		for i in range(1,num):
			lines.append(file.readline().strip().split("|"))
		return lines
	else:
		return line

##################################################################
# Programs
start_time = time.time()

with open("Data/trainData.txt","w") as trainData:
	with open("RawData/trainIdx2.txt") as trainFile:
		for line in trainFile:
			if "|" in line:
				cur_user = line.split("|")[0]
				print(cur_user,end="\r")
			else:
				trainData.write(cur_user+"\t"+line)

print("----------------------------------------------------------------")
print("Rewrite train data finished, Spend %.2f s"%(time.time()-start_time))
print("----------------------------------------------------------------")

with open("Data/testData.txt","w") as testData:
	with open("RawData/testIdx2.txt") as testFile:
		for line in testFile:
			if "|" in line:
				cur_user = line.split("|")[0]
				print(cur_user,end="\r")
			else:
				testData.write(cur_user+"\t"+line)

print("Rewrite test data finished, Spend %.2f s"%(time.time()-start_time))
print("----------------------------------------------------------------")
print("Start Spark")
print("----------------------------------------------------------------")

sparkC = SparkContext()
sqlC = SQLContext(sparkC)
trainData = sparkC.textFile("Data/trainData.txt").map(lambda line: line.split("\t"))
testData = sparkC.textFile("Data/testData.txt").map(lambda line: line.split("\t"))

trainDataFrame = sqlC.createDataFrame(trainData,["user","item","rating"])
testDataFrame = sqlC.createDataFrame(testData,["user","item"])

# You can change the rank and maxIter here
als = ALS(rank = 10,maxIter = 20)

model = als.fit(trainDataFrame)

predTestData = model.transform(testDataFrame)

prediction = sorted(predTestData.collect(), key = lambda r: int(r[0]))

with open("Results/raw_prediction.txt","w") as predFile:
	for line in prediction:
		if line[2]!=line[2]:
			temp_str = "0"
		else:
			temp_str = str(int(line[2]))
		predFile.write(str(line[0])+"|"+str(line[1])+"|"+temp_str+"\n")

sparkC.stop()

print("----------------------------------------------------------------")
print("Spark predicting job finished, Spend %d s"%(time.time()-start_time))

print("----------------------------------------------------------------")
print("Start to reorder prediction")
print("----------------------------------------------------------------")

temp_dic = {}
with open("Results/prediction.txt","w") as predFile:
	with open("Results/raw_prediction.txt") as rawFile:
		with open("RawData/testIdx2.txt") as testFile:
			raw_lines = read_lines(rawFile,6)
			while raw_lines:
				temp_dic.clear()
				user_median = np.median([int(x[2]) for x in raw_lines])
				for row in raw_lines:
					if int(row[2])>=user_median:
						temp_dic[str(row[1])]="1"
					else:
						temp_dic[str(row[1])]="0"
				test_line = testFile.readline()
				for i in range(6):
					test_line = testFile.readline().strip()
					predFile.write(temp_dic[test_line]+"\n")
				raw_lines = read_lines(rawFile,6)

print("----------------------------------------------------------------")
print("Reorder prediction finished, Spend %d s"%(time.time()-start_time))
