# 🐞 ML Bug Risk Prediction Project

This project demonstrates a complete, step-by-step machine learning workflow for predicting high-risk software bugs using historical bug data and component risk scoring. By leveraging data-driven analytics and ML models, QA teams can proactively identify and prioritize bugs that are most likely to impact product quality.

---

## 📁 Project Structure

- **data/**
  - **raw/**: Raw bug data files.
  - **processed/**: Cleaned and transformed data for model training.
- **notebooks/**
  - **Bug Risk Prediction Demo - Step-by-Step ML Workflow.ipynb**: Main notebook showing the full ML workflow, from data loading to model interpretation.
  - **exploratory_analysis.ipynb**: Exploratory data analysis and visualization.
- **src/**
  - **data_preprocessing.py**: Data loading and cleaning functions.
  - **feature_engineering.py**: Feature creation and transformation.
  - **model_training.py**: Model training scripts.
  - **model_evaluation.py**: Model evaluation utilities.
  - **predict.py**: Prediction scripts for new data.
- **tests/**: Unit tests for each module.
- **requirements.txt**: Python dependencies.
- **.gitignore**: Files and folders to ignore in version control.

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone git@github.com:QAPournima/ML-Predicting-Bugs.git
   cd ML-Predicting-Bugs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place raw bug data in `data/raw/`.
   - Run preprocessing scripts or use the notebook to generate processed data.

4. **Run the notebook:**
   - Open `notebooks/Bug Risk Prediction Demo - Step-by-Step ML Workflow.ipynb` in Jupyter or VS Code.
   - Follow the workflow cells to train, evaluate, and interpret the bug risk prediction model.

---

## 🧠 ML Workflow Overview

The main notebook walks through these steps:

1. **Data Loading:**  
   Loads historical bug data from CSV files.

2. **Feature Engineering:**  
   Calculates risk scores for each bug/component and creates one-hot encoded features.

3. **Data Preparation:**  
   Combines features, defines the target variable (high risk or not), and splits data into training and test sets.

4. **Model Training:**  
   Trains Random Forest and XGBoost classifiers to predict high-risk bugs.

5. **Prediction:**  
   Uses trained models to predict risk on test data.

6. **Evaluation:**  
   Assesses model performance using accuracy, precision, recall, F1-score, and confusion matrix.

7. **Results Visualization:**  
   Visualizes feature importance and risk distribution.

8. **Interpretability:**  
   Explains which components are most associated with high risk and provides human-readable explanations for predictions.

---

## 📊 Example: Bug Risk Analysis Report

- **Component Risk Scoring:**  
  Components are ranked by historical bug frequency and assigned risk scores (high, moderate, low).
- **Visualizations:**  
  Pie charts, summary tables, and risk scoring tables highlight high-risk areas.
- **Actionable Insights:**  
  Focus QA and development efforts on components most likely to cause issues.

---

## 💡 How This Supports Predictive QA

- **Feature Engineering:**  
  Risk scores and component histories become features for ML models.
- **Continuous Learning:**  
  As new bugs are reported, retrain models for improved accuracy.
- **Prioritization:**  
  Automatically flag and prioritize high-risk bugs for review and testing.

---

## 📚 Usage Examples

- **Preprocess data:**
  ```python
  from src.data_preprocessing import preprocess_data
  preprocess_data('data/raw/bugs.csv')
  ```

- **Train the model:**
  ```python
  from src.model_training import train_model
  train_model('data/processed/processed_data.csv')
  ```

- **Evaluate the model:**
  ```python
  from src.model_evaluation import evaluate_model
  evaluate_model('data/processed/processed_data.csv')
  ```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License.

---

**For a full end-to-end example, see the notebook:**  
`notebooks/Bug Risk Prediction Demo - Step-by-Step ML Workflow.ipynb`
