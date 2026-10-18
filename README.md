# Credit Card Fraud Detection

Machine learning project to detect fraudulent credit card transactions on a highly imbalanced dataset.

## Dataset
[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions, of which only 492 (0.17%) are fraud. The dataset is not included in this repo because of its size.

## Approach
1. Exploratory data analysis
2. Scaled `Amount` and `Time` with RobustScaler
3. Stratified 80/20 train-test split
4. Handled class imbalance with SMOTE (training data only) and class weighting
5. Trained Logistic Regression, Logistic Regression + SMOTE, Random Forest and XGBoost
6. Evaluated with precision, recall and PR-AUC rather than accuracy
7. Explained predictions with SHAP and tuned the decision threshold

## Results
| Model | PR-AUC (Average Precision) |
|---|---|
| XGBoost | 0.88 |
| Random Forest | 0.86 |
| Logistic Regression | 0.74 |
| Logistic Regression + SMOTE | 0.72 |

**Best model: XGBoost.** On the held-out test set (56,962 transactions, 98 frauds) at threshold 0.5, it caught 83 of 98 frauds (84.7% recall) with 87.4% precision, raising only 12 false alarms among 56,864 normal transactions.


### Threshold trade-off (XGBoost)
| Threshold | Precision | Recall | Frauds caught | False alarms |
|---|---|---|---|---|
| 0.3 | 85.6% | 84.7% | 83/98 | 14 |
| 0.5 | 87.4% | 84.7% | 83/98 | 12 |
| 0.7 | 88.2% | 83.7% | 82/98 | 11 |
| 0.9 | 91.0% | 82.7% | 81/98 | 8 |

The model's fraud scores are well separated, so raising the threshold to 0.9 cuts false alarms from 14 to 8 while losing only 2 caught frauds. The remaining missed frauds get very low probabilities, which is a limit of the features rather than the threshold.

## Visuals

![Confusion matrix](confusion_matrix.png)
![PR curves](pr_curves.png)
![SHAP](shap_plot.png)
![App: fraud flagged](app_fraud.png)
![App: normal](app_normal.png)


## Key insights
- Accuracy is misleading here: predicting "not fraud" every time scores about 99.8%, so models were compared on PR-AUC, precision and recall.
- Tree-based models (XGBoost, Random Forest) clearly outperformed Logistic Regression, even when SMOTE was applied to the linear model.
- SHAP shows the most influential features are V14, V4, V12, V10 and V8. Low values of V14 and V12 and high values of V4 push predictions toward fraud.
- Lowering the decision threshold catches more fraud but raises false alarms, a trade-off the demo app lets you explore with a slider.

- - Raising the decision threshold from 0.3 to 0.9 raised precision from 85.6% to 91.0% while recall only fell from 84.7% to 82.7%.

## Run the demo
```
pip install -r requirements.txt
streamlit run app.py
```
