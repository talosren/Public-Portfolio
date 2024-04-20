# Classification and Regression Tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split # for data spliting
from sklearn.tree import DecisionTreeClassifier, plot_tree # for regression tree
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score # for mode evaluation

from sklearn.linear_model import LogisticRegression # for logit models
import statsmodels.api as sm

df = pd.read_csv('data/Stevens.csv')

print(df.shape) # (566 Rows, 9 Columns)
print(df.info)

print(df['Petitioner'].unique())
# ['BUSINESS' 'CITY' 'EMPLOYEE' 'AMERICAN.INDIAN' 'INJURED.PERSON'
#  'GOVERNMENT.OFFICIAL' 'OTHER' 'STATE' 'US' 'CRIMINAL.DEFENDENT'
#  'EMPLOYER' 'POLITICIAN']

# Creating Petitioner Categories Distribution
sns.countplot(x = 'Petitioner', data = df)
plt.title('Distribution of Petitioner Categories')
plt.xlabel('Petitoner')
plt.ylabel('Count')

plt.xticks(rotation = 75)
plt.tight_layout()
# plt.show()

# Splitting dataframe into independent and dependent variables

X = df.drop(['Docket', 'Term', 'Reverse'], axis = 1)
y = df['Reverse']
X = pd.get_dummies(X)

# Create training/testing data, building CART model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CART model with minimum sample size 40
leaves = 40
dt = DecisionTreeClassifier(min_samples_leaf = leaves)
dt.fit(X_train, y_train)

plt.figure(figsize = (16, 8))
plot_tree(dt, feature_names= X.columns, class_names = ['Affirm', 'Reverse'], filled = True)
plt.title('Decision Tree with Minimum Sample Size 40')

# Define CART model with minimum sample size 80
leaves = 80
dt2 = DecisionTreeClassifier(min_samples_leaf = leaves)
dt2.fit(X_train, y_train)

plt.figure(figsize = (16, 8))
plot_tree(dt2, feature_names= X.columns, class_names = ['Affirm', 'Reverse'], filled = True)
plt.title('Decision Tree with Minimum Sample Size 80')
# plt.show()

y_pred = dt.predict(X_test)
print(y_pred)

y_pred_proba = dt.predict_proba(X_test)
print(y_pred_proba[:10])

print(pd.crosstab(y_test, y_pred))

# Accuracy Scores
print(accuracy_score(y_test, y_pred))

y_pred_80 = dt2.predict(X_test)
print(f'The overall accuracy is {accuracy_score(y_test, y_pred_80)}')

# ROC and AUC

# Predict probabilities on the test set
# Probabilities for the positive class
y_probs = dt.predict_proba(X_test)[:, 1]

# Compute TPR and FPR
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, color = 'blue', lw = 2, label = f'ROC curve (AUC={roc_auc:0.4f})')
plt.plot([0, 1], [0, 1], color = 'darkgray', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Finding the best threshold
# Recalculate for convenience
roc_auc_test = roc_auc_score(y_test, y_probs)

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

# Calculate the F1 score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_threshold_idx]

print(f"ROC AUC Train: {roc_auc:.4f}")
print(f"ROC AUC Test: {roc_auc:.4f}")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Precision: {precision[best_threshold_idx]:.4f}")
print(f"Recall: {recall[best_threshold_idx]:.4f}")
print(f"F1 Score: {f1_scores[best_threshold_idx]:.4f}")