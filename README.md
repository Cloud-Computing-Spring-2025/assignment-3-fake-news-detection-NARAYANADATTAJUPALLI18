#!/bin/bash

# Fake News Classification - Project

## Overview

This project aims to classify news articles as either "REAL" or "FAKE" using machine learning techniques. The project uses PySpark for data processing, model training, and evaluation. The dataset contains synthetically generated articles to simulate real-world classification tasks.

## Project Structure

\`\`\`
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
\`\`\`

## Prerequisites

To run this project, you need to have the following installed:

- **Apache Spark** (version 3.x or higher)
- **PySpark** (version 3.x or higher)
- **Python** (version 3.x or higher)
- **Java** (version 8 or higher, for Spark compatibility)

### Installing Dependencies

You can install the necessary Python dependencies using pip:

\`\`\`bash
pip install pyspark
\`\`\`

## Project Workflow

### Step 1: Data Preparation

The dataset \`fake_news_sample.csv\` is generated synthetically. It contains 500 rows of text, with 250 labeled as "REAL" and 250 labeled as "FAKE". This dataset is used to train the model.

### Step 2: Feature Engineering (Task 3)

The data is preprocessed and transformed into features suitable for machine learning. This step outputs a parquet file (\`task3_output.parquet\`), which is used in the subsequent steps.

### Step 3: Model Training (Task 4)

The **Logistic Regression** model is trained using the processed dataset from Task 3. Cross-validation is used to evaluate the model's performance, and regularization (\`regParam=0.1\`) is applied to avoid overfitting.

\`\`\`bash
spark-submit task4.py
\`\`\`

- This will train the model and save predictions in \`task4_output.csv\`.
- Misclassified rows will also be shown in the console for debugging purposes.

### Step 4: Model Evaluation (Task 5)

After training, the model is evaluated using **AUC-ROC** and **AUC-PR** metrics, which are more reliable in binary classification tasks.

\`\`\`bash
spark-submit task5.py
\`\`\`

- This will generate the evaluation metrics and save the results in \`task5_output.csv\`.

### Step 5: Results

The metrics are saved in CSV format with the following columns:

- **AUC-ROC**: The area under the Receiver Operating Characteristic curve.
- **AUC-PR**: The area under the Precision-Recall curve.

### Example Output:
\`\`\`bash
+------------+-----+
| Metric     | Value|
+------------+-----+
| AUC-ROC    | 1.0 |
| AUC-PR     | 1.0 |
+------------+-----+
\`\`\`

## Additional Notes

- **Data Leakage**: The dataset is synthetic, and we made sure there is no label leakage. However, for real-world datasets, it's important to ensure no leakage between the training and test sets.
- **Imbalance**: The dataset is balanced with 250 "REAL" and 250 "FAKE" articles. For real-world applications, datasets may need balancing techniques such as oversampling or undersampling.
- **Shuffling Labels**: In case of overfitting, you can shuffle the labels and observe how well the model generalizes.

## Conclusion

This project demonstrates how to train and evaluate a fake news detection model using PySpark. With the adjustments made, the model is less likely to overfit on clean, synthetic data, and more robust evaluation metrics are used to validate performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOL

echo "README.md file has been created!"
