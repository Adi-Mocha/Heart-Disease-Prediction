# Heart Disease Prediction

**Course:** Predictive Analytics (CSE3141), B.Tech CSE V Sem, Manipal University Jaipur
**Name:** Aditya Singh | **Reg. No.:** 2427030054

## Problem Statement
Predict whether a patient has heart disease from clinical measurements (age, blood pressure,
cholesterol, ECG results, exercise test results, etc.). This is a binary classification problem.

## Dataset
UCI Heart Disease (Cleveland subset): https://archive.ics.uci.edu/dataset/45/heart+disease
303 patients, 13 features. Target `num` (0-4) converted to 0 = no disease, 1 = disease.
(The same data is also mirrored on Kaggle: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

## Machine Learning Approach
1. **Data cleaning:** missing values in `ca` and `thal` imputed inside the pipeline (no leakage); outliers checked with the IQR rule and box plots
2. **EDA:** mean/median/mode, variance, standard deviation, z-scores, bar chart, pie chart, histograms, heatmap, scatter plot
3. **Feature engineering:** heart-rate ratio, high-cholesterol flag, high-BP flag; scaling and one-hot encoding
4. **Feature selection:** mutual information ranking and `SelectKBest` (k tuned by grid search)
5. **Models:** Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA)
6. **Validation:** stratified 80/20 train-test split, stratified 5-fold cross-validation
7. **Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC
8. **Hyperparameter tuning:** Grid Search (C, shrinkage, reg_param, number of features)
9. **Overfitting check:** train vs test AUC gap and learning curves
10. **Interpretation:** logistic regression coefficients and odds ratios

## Results
Test-set results (from `reports/results.csv` after running the notebook):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.852 | 0.806 | 0.893 | 0.847 | 0.942 |
| LDA | 0.902 | 0.867 | 0.929 | 0.897 | 0.952 |
| QDA | 0.902 | 0.867 | 0.929 | 0.897 | 0.940 |
## Conclusion
LDA performed best on the held-out test set (accuracy 0.902, F1 0.897, AUC-ROC 0.952), slightly ahead of Logistic Regression and QDA. All three models achieved high recall (≥ 0.89), which is important in a medical setting where missing a diseased patient is costly. Because the test set contains only 61 patients, the differences between models are small and may not be statistically significant. Limitations: the dataset is small (303 patients) and comes from a single hospital, so results may not generalise. The model should support doctors, not replace them.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/heart_disease_analysis.ipynb
```
Run all cells. Figures are saved to `reports/figures/`.

## Syllabus Topics Covered
- Data preparation and exploration (EDA, outliers, missing values, feature engineering/selection)
- Logistic regression and inference (logistic regression, LDA, QDA)
- Model validation and optimization (cross-validation, F1, AUC-ROC, grid search, overfitting)
- Ethical considerations in predictive modelling

## Project Structure
```
heart-disease-prediction/
├── README.md
├── requirements.txt
├── data/
├── notebooks/heart_disease_analysis.ipynb
└── reports/figures/
```
