# task5.py - Evaluation with Binary Metrics
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("FakeNewsEvaluationUpdated").getOrCreate()

df = spark.read.csv("task4_output.csv", header=True, inferSchema=True)

evaluator_acc = BinaryClassificationEvaluator(
    labelCol="label_index", rawPredictionCol="prediction", metricName="areaUnderROC")

evaluator_f1 = BinaryClassificationEvaluator(
    labelCol="label_index", rawPredictionCol="prediction", metricName="areaUnderPR")

roc_auc = evaluator_acc.evaluate(df)
pr_auc = evaluator_f1.evaluate(df)

# Create DataFrame for output
results = spark.createDataFrame([
    ("AUC-ROC", roc_auc),
    ("AUC-PR", pr_auc)
], ["Metric", "Value"])

results.show()

results.write.mode("overwrite").csv("task5_output.csv", header=True)
