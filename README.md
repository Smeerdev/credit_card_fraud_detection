# Sampling and Classification in Imbalanced Datasets

This repository contains a Jupyter Notebook that addresses the challenges of handling imbalanced datasets using sampling techniques and evaluates different classification models. The notebook focuses on oversampling with SMOTE and generates different data samples for comparative evaluation of models.

## Features

1. **Dataset Overview**: 
   - The dataset contains 772 entries with 30 features (`V1` to `V28`), an `Amount` column, and a `Class` column (target variable).
   - The `Class` column is imbalanced, with 763 entries labeled as `0` (non-fraudulent) and only 9 entries labeled as `1` (fraudulent).

2. **Steps in the Notebook**:
   - **Imbalance Check**: Identify the level of class imbalance in the dataset.
   - **Oversampling with SMOTE**: Apply SMOTE to create a balanced dataset.
   - **Sampling Techniques**: Generate five distinct samples:
     - Random sampling with different seeds.
     - Stratified sampling.
     - Using the first 200 rows.
     - Using the last 200 rows.
   - **Model Training**: Train and evaluate the following models:
     - Gradient Boosting Classifier
     - Naive Bayes
     - Linear Discriminant Analysis (LDA)
     - Multi-Layer Perceptron (MLP) Classifier
     - Ridge Classifier
   - **Performance Evaluation**: Evaluate models using accuracy scores on the sampled data.

3. **Output**: Each sample's results are presented with model accuracy, ensuring a comprehensive comparison.

## How to Use

1. **Prerequisites**:
   - Python 3.x
   - Required libraries:
     - `pandas`
     - `imblearn` (for SMOTE)
     - `scikit-learn`

2. **Steps**:
   - Load the dataset (`Creditcard_data.csv`).
   - Execute the notebook to generate balanced samples, save them, and evaluate models.

3. **Outputs**:
   - Sample datasets saved as `sample1.csv`, `sample2.csv`, ..., `sample5.csv`.
   - Classification results for each sample and model displayed in the notebook.

## Key Insights

- SMOTE successfully balances the dataset, allowing better evaluation of classifiers.
- Different sampling techniques impact model performance, highlighting the importance of data preparation.
- Gradient Boosting consistently performs well, but model choice can depend on the specific use case and dataset characteristics.

## Limitations

- The notebook is tailored to a specific dataset and might require modifications for other datasets.
- Further hyperparameter tuning and evaluation metrics like F1 score, precision, and recall are recommended for better insights.

## License

This project is open-source and available under the MIT License.
