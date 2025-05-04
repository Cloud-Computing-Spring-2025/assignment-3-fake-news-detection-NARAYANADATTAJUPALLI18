from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover

spark = SparkSession.builder.appName("FakeNewsTextProcessing").getOrCreate()

df = spark.read.csv("task1_output.csv", header=True, inferSchema=True)

# Preprocess text
df_cleaned = df.withColumn("text", lower(col("text")))
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_words = tokenizer.transform(df_cleaned)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_words)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Convert array of words to string for CSV output
array_to_str = udf(lambda words: ' '.join(words) if words else '', StringType())
df_task2 = df_filtered.withColumn("filtered_words_str", array_to_str(col("filtered_words")))
df_task2.select("id", "title", "filtered_words_str", "label") \
        .write.mode("overwrite").csv("task2_output.csv", header=True)