# IBM HR Analytics — Employee Attrition 

## Summary (what this project does)
This project analyzes the IBM HR Employee Attrition dataset and builds predictive models to estimate whether an employee is likely to leave the company (**Attrition = Yes**). The intent is to help HR teams prioritize *proactive* retention actions (career development conversations, manager support, workload balancing, compensation review, etc.).

## Key findings (high-level)
- Attrition is a **minority class**, so **accuracy alone is misleading**. The notebook evaluates models using **ROC-AUC**, **Recall**, **F1**, and **PR-AUC (Average Precision)**.
- Multiple models are compared using **Stratified K-Fold cross-validation**, then tuned using **GridSearchCV**.
- The final model is evaluated once on a **holdout test set** and interpreted with **permutation importance** (and coefficients if logistic regression is selected).

> Note: Exact metric values will appear in the notebook output after you run it (they depend on random split + tuning).

## Project structure
```
.
├── data/
│   └── IBM-HR-Employee-Attrition.csv
├── IBM_HR_Attrition_Capstone.ipynb
└── README.md
```

## Notebook
- **Notebook:** `IBM_HR_Attrition_Capstone.ipynb`

## How to run
1. Put the dataset CSV in `./data/` (the notebook also searches common filenames in the project root).
2. Open and run the notebook top-to-bottom.

### Environment
The notebook uses common Python ML libraries:
- pandas, numpy, matplotlib
- scikit-learn

## What’s inside the notebook
- Business framing and EDA
- Data quality checks + feature engineering
- Baseline logistic regression
- Pipeline-based preprocessing (imputation + scaling + one-hot encoding)
- Cross-validated model comparison (LogReg, RandomForest, HistGradientBoosting)
- Hyperparameter tuning with GridSearchCV
- Holdout test evaluation (confusion matrix, ROC curve, PR curve)
- Model interpretation 
- Recommendations and next steps 
