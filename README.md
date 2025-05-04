
# Fake News Classification - Project

## Overview

This project aims to classify news articles as either "REAL" or "FAKE" using machine learning techniques. The project uses PySpark for data processing, model training, and evaluation. The dataset contains synthetically generated articles to simulate real-world classification tasks.

## Project Structure

```
/project-directory
    ├── data
    │   └── fake_news_sample.csv        # Synthetic dataset containing news articles
    ├── task1.py                        # Data Preprocessing: Initial data cleaning and preparation
    ├── task2.py                        # Feature Engineering: Extracting features from text data
    ├── task3.py                        # Feature Transformation: Transforming features for modeling
    ├── task4.py                        # Model Training with Cross-Validation and Regularization
    ├── task5.py                        # Model Evaluation with AUC-ROC and AUC-PR metrics
    ├── task1_output.csv                # Output of data preprocessing (cleaned data)
    ├── task2_output.parquet            # Output of feature extraction (features for modeling)
    ├── task3_output.parquet            # Output of feature transformation (final features for model training)
    ├── task4_output.csv                # Predictions after model training
    ├── task5_output.csv                # Evaluation metrics (AUC-ROC, AUC-PR)
    └── README.md                       # Project documentation
```

## Prerequisites

To run this project, you need to have the following installed:

- **Apache Spark** (version 3.x or higher)
- **PySpark** (version 3.x or higher)
- **Python** (version 3.x or higher)
- **Java** (version 8 or higher, for Spark compatibility)

### Installing Dependencies

You can install the necessary Python dependencies using pip:

```bash
pip install pyspark
```

## Project Workflow

### Step 1: Data Preparation (Task 1)

The dataset \`fake_news_sample.csv\` is generated synthetically. It contains 500 rows of text, with 250 labeled as "REAL" and 250 labeled as "FAKE". This dataset is used to train the model.

**Task 1 Script**: `task1.py` performs initial data preprocessing to clean and prepare the data.

Run the preprocessing step with:

```bash
spark-submit task1.py
```

This will generate the cleaned dataset in \`task1_output.csv\`.

### Step 2: Feature Engineering (Task 2)

In this step, features are extracted from the text data using various text processing techniques such as tokenization and stopword removal. The output is stored in a Parquet file for efficient storage.

**Task 2 Script**: `task2.py` performs feature extraction on the cleaned dataset.

Run the feature engineering step with:

```bash
spark-submit task2.py
```

This will generate the output in \`task2_output.parquet\`.

### Step 3: Feature Transformation (Task 3)

After feature extraction, the data is transformed into a format suitable for training the machine learning model. This transformation involves vectorizing text data and indexing labels.

**Task 3 Script**: `task3.py` performs feature transformation and outputs a Parquet file with the final features.

Run the feature transformation step with:

```bash
spark-submit task3.py
```

This will generate the output in \`task3_output.parquet\`.

### Step 4: Model Training (Task 4)

The **Logistic Regression** model is trained using the processed dataset from Task 3. Cross-validation is used to evaluate the model's performance, and regularization (\`regParam=0.1\`) is applied to avoid overfitting.

**Task 4 Script**: `task4.py` trains the model and saves the predictions.

Run the model training step with:

```bash
spark-submit task4.py
```

- This will train the model and save predictions in \`task4_output.csv\`.
- Misclassified rows will also be shown in the console for debugging purposes.

### Step 5: Model Evaluation (Task 5)

After training, the model is evaluated using **AUC-ROC** and **AUC-PR** metrics, which are more reliable in binary classification tasks.

**Task 5 Script**: `task5.py` evaluates the trained model using the AUC-ROC and AUC-PR metrics.

Run the evaluation step with:

```bash
spark-submit task5.py
```

- This will generate the evaluation metrics and save the results in \`task5_output.csv\`.

### Step 6: Results

The metrics are saved in CSV format with the following columns:

- **AUC-ROC**: The area under the Receiver Operating Characteristic curve.
- **AUC-PR**: The area under the Precision-Recall curve.

### Example Output:
```bash
+------------+-----+
| Metric     | Value|
+------------+-----+
| AUC-ROC    | 0.89 |
| AUC-PR     | 0.88 |
+------------+-----+
```

## Additional Notes

- **Data Leakage**: The dataset is synthetic, and we made sure there is no label leakage. However, for real-world datasets, it's important to ensure no leakage between the training and test sets.
- **Imbalance**: The dataset is balanced with 250 "REAL" and 250 "FAKE" articles. For real-world applications, datasets may need balancing techniques such as oversampling or undersampling.
- **Shuffling Labels**: In case of overfitting, you can shuffle the labels and observe how well the model generalizes.

## Conclusion

This project demonstrates how to train and evaluate a fake news detection model using PySpark. With the adjustments made, the model is less likely to overfit on clean, synthetic data, and more robust evaluation metrics are used to validate performance.


