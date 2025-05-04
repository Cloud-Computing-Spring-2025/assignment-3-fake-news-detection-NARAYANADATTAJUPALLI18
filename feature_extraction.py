from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.feature import Tokenizer

spark = SparkSession.builder.appName("FakeNewsFeatureExtraction").getOrCreate()

df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

tokenizer = Tokenizer(inputCol="filtered_words_str", outputCol="filtered_words")
df_tokenized = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized_data = hashingTF.transform(df_tokenized)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
data_with_labels = indexer.fit(rescaled_data).transform(rescaled_data)

df_task3 = data_with_labels.select("id", "filtered_words", "features", "label_index")
df_task3.write.mode("overwrite").parquet("task3_output.parquet")
