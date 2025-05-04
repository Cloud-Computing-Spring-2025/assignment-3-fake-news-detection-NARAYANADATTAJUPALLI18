# task4.py - Model Training with CrossValidation and Regularization
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("FakeNewsModelTrainingCV").getOrCreate()

df = spark.read.parquet("task3_output.parquet")

# Logistic Regression with regularization
lr = LogisticRegression(featuresCol="features", labelCol="label_index", regParam=0.1)

# CrossValidator setup
evaluator = BinaryClassificationEvaluator(labelCol="label_index")
paramGrid = ParamGridBuilder().build()

cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(df)
predictions = cv_model.transform(df)

# Save predictions to CSV
predictions.select("id", "label_index", "prediction") \
    .write.mode("overwrite").csv("task4_output.csv", header=True)

# Show misclassified rows (optional debugging)
print("Misclassified rows:")
predictions.filter("label_index != prediction").show()
