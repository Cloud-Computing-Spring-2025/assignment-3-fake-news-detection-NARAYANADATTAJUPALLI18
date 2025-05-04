from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FakeNewsTask1").getOrCreate()

df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

df.show(5)
print("Total articles:", df.count())
df.select("label").distinct().show()

df.write.mode("overwrite").csv("task1_output.csv", header=True)