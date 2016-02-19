from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
sparkC = SparkContext()
sqlC = SQLContext(sparlC)
data_list = [(0,0,4.0),(0,1,2.0),(1,1,3.0),(1,2,4.0),(2,1,1.0),(2,2,5.0)]
dataFrame = sqlC.createDataFrame(data_list,["user","item","rating"])
als = ALS(rank=10,maxIter=5)
model = als.fit(dataFrame)
testData = sqlC.createDataFrame([(0,2),(1,0),(2,0)],["user","item"])
predicions = sorted(model.transform(test).collect(),key = lambda r: r[0])
