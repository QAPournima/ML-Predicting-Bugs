# Machine Learning Bug Prediction Project

This project aims to implement a machine learning model to predict bugs in software projects. The model will analyze historical data to identify patterns and potential bug occurrences.

## Project Structure

- **data/**
  - **raw/**: Contains raw data files used for training and testing the model.
  - **processed/**: Stores processed data files that have been cleaned and transformed for model training.
  
- **notebooks/**
  - **exploratory_analysis.ipynb**: Jupyter notebook for exploratory data analysis, visualizing data distributions, and understanding relationships between features.

- **src/**
  - **data_preprocessing.py**: Functions for loading and preprocessing the raw data, including cleaning and handling missing values.
  - **feature_engineering.py**: Functions for creating new features from existing data to improve model performance.
  - **model_training.py**: Responsible for training the machine learning model using processed data and saving the trained model.
  - **model_evaluation.py**: Functions for evaluating the trained model's performance using metrics such as accuracy, precision, and recall.
  - **predict.py**: Functions for making predictions using the trained model on new data.

- **tests/**
  - **test_data_preprocessing.py**: Unit tests for the functions defined in `data_preprocessing.py`.
  - **test_feature_engineering.py**: Unit tests for the functions defined in `feature_engineering.py`.
  - **test_model_training.py**: Unit tests for the functions defined in `model_training.py`.

- **requirements.txt**: Lists all required Python libraries and their versions needed to run the project.

- **.gitignore**: Specifies files and directories that should be ignored by Git, such as data files and temporary files.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-bug-prediction
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place raw data files in the `data/raw/` directory.
   - Run the preprocessing script to clean and transform the data.

4. Train the model:
   - Use the `model_training.py` script to train the model with the processed data.

5. Evaluate the model:
   - Use the `model_evaluation.py` script to assess the model's performance.

6. Make predictions:
   - Use the `predict.py` script to make predictions on new data.

## Usage Examples

- To preprocess the data:
  ```python
  from src.data_preprocessing import preprocess_data
  preprocess_data('data/raw/data_file.csv')
  ```

- To train the model:
  ```python
  from src.model_training import train_model
  train_model('data/processed/processed_data.csv')
  ```

- To evaluate the model:
  ```python
  from src.model_evaluation import evaluate_model
  evaluate_model('data/processed/processed_data.csv')
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

# 📊 Bug Risk Analysis Report: DemoData (APP)

## Overview

This project demonstrates how to use data-driven analytics and machine learning to **predict software bugs before they happen**. By analyzing historical bug data, we identify high-risk components and patterns, enabling proactive quality assurance and targeted testing. The ultimate goal is to reduce the number and severity of bugs in production by focusing resources where they are needed most.

---

## How the Report Works

### 1. Data Collection and Preparation

- Bug data is exported from a bug tracking system (such as Jira).
- Only bugs created or resolved in the **last 6 months** are considered, ensuring the analysis is relevant to current development and usage trends.
- Each bug is associated with one or more software components.

### 2. Component Risk Scoring

- The frequency of bugs per component is calculated.
- Components are ranked by the number of bugs they have accumulated in the last 6 months.
- A **risk score** is assigned to each component based on its rank:
  - **High risk**: Components with the most bugs.
  - **Moderate risk**: Components with an average number of bugs.
  - **Low risk**: Components with few or no recent bugs.

### 3. Visualizations and Summaries

- **Pie Chart**:  
  Shows the distribution of bugs across components, highlighting which areas of the application are most problematic.
- **Summary Section**:  
  Presents key metrics such as the total number of bugs, the top risk component, and the criteria for scoring.
- **Risk Scoring Table**:  
  Lists the top 10 bugs/components by risk score, with color-coded risk levels and explanations for each score.

---

## Key Insights from This Report

- **CallHistory** is currently the highest-risk component, accounting for 42.2% of recent bugs.
- The top 10 components are responsible for the majority of bugs, suggesting that targeted interventions in these areas could yield significant quality improvements.
- The risk scoring table provides actionable explanations, helping developers and QA teams understand why certain components are considered high or low risk.

---

## How This Supports Machine Learning for Bug Prediction

- **Feature Engineering**:  
  The risk scores and component histories generated here can be used as features in a machine learning model. For example, a model could learn that bugs in high-risk components are more likely to be severe or to recur.
- **Training Data**:  
  The labeled data (bugs, components, risk scores) can be used to train classification or regression models to predict the likelihood of future bugs.
- **Feedback Loop**:  
  As new bugs are reported and resolved, the report can be regenerated, providing up-to-date data for retraining models and refining predictions.
- **Prioritization**:  
  By identifying high-risk components, teams can focus code reviews, testing, and monitoring on the areas most likely to cause issues, thus preventing bugs before they reach users.

---

## 🧠 Integrating Risk Scoring with Machine Learning for Predictive Bug Analytics

### 1. Feature Engineering

The outputs from this report—such as component risk scores, bug frequencies, and component histories—can be used as features for machine learning models. Here’s how you can extract and use them:

- **Component Risk Score**: Use the computed risk score for each component as a numerical feature.
- **Bug Frequency**: Use the count of bugs per component or per time window as a feature.
- **Component Name (One-Hot Encoding)**: Convert component names to one-hot encoded vectors or embeddings.
- **Bug Metadata**: Include other bug fields (e.g., severity, reporter, time to resolve) as features.

**Example:**
```python
# Example: Creating a feature set for ML
features = summary_df.copy()
features['Component_Risk_Score'] = features['Risk Score'].apply(lambda x: int(str(x).split()[0]))
# Add more features as needed
```

### 2. Preparing the Training Data

- **Labeling**:  
  - For classification: Label bugs as "likely to reoccur" or "not likely", or "high risk" vs "low risk".
  - For regression: Use the time to next bug, or bug severity, as the target variable.
- **Merging Data**:  
  - Merge risk scores and bug features with historical bug data to create a comprehensive dataset.

**Example:**
```python
# Suppose you have a historical bug dataframe `bugs`
ml_data = bugs.merge(features[['🐞 Bug ID', 'Component_Risk_Score']], left_on='key', right_on='🐞 Bug ID', how='left')
# Add your label column, e.g., 'is_critical'
```

### 3. Model Selection and Training

- **Choose a Model**:  
  - For classification: Logistic Regression, Random Forest, XGBoost, etc.
  - For regression: Linear Regression, Gradient Boosting, etc.
- **Train/Test Split**:  
  - Split your data into training and test sets.
- **Model Training**:  
  - Train the model using your engineered features.

**Example:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = ml_data[['Component_Risk_Score', ...]]  # Add other features
y = ml_data['is_critical']  # Your label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 4. Prediction and Deployment

- **Prediction**:  
  - Use the trained model to predict the risk of new bugs or components.
- **Integration**:  
  - Integrate the model into your bug triage or CI/CD pipeline to flag high-risk bugs/components in real time.

**Example:**
```python
# Predict risk for new bugs
predictions = model.predict(X_test)
```

### 5. Continuous Learning

- **Feedback Loop**:  
  - Regularly update the risk scores and retrain the model as new bug data becomes available.
- **Automation**:  
  - Automate the data extraction, feature engineering, and model retraining process.

---

## Conclusion

This report is a practical example of how data-driven analysis and machine learning can be integrated into the software development lifecycle. By continuously monitoring bug trends and component risk, organizations can move from reactive bug fixing to proactive bug prevention, ultimately delivering more reliable software to users.

---

**Next Steps:**  
- Integrate these risk scores and component histories into your machine learning pipeline.
- Use the insights to guide test automation, code review, and resource allocation.
- Continuously update the report and retrain models as new data becomes available.

---

*For a sample end-to-end notebook or code for a specific ML use case, see the `ml-bug-prediction/notebooks/Demo.ipynb` file in this repository.*