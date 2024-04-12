import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

df = pd.read_csv('data/Quality.csv')

print(df.head(5))
print(df.columns)
print(df.info())

# Column Names + Info
# 'MemberID' (int), 'ERVisits' (int), 'OfficeVisits' (int),
# 'Narcotics' (int), 'ProviderCount' (int), 'NumberClaims' (int),
# 'StartedOnCombination' (str), 'PoorCare' (int)

# Creating a subplot with 3 rows and 3 columns
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (12,8))
axes = axes.ravel()

variables = ['ERVisits', 'OfficeVisits', 'Narcotics', 'ProviderCount', 'NumberClaims', 'StartedOnCombination', 'PoorCare']
labels = ['ER Visits', 'Office Visits', 'Narcotics', 'Provider Count', 'Number of Claims', 'Started On Combination', 'Poor Care']

# Create histograms

# Create histograms for each variable
for i, var in enumerate(variables):
    ax = axes[i]

    if var in ['StartedOnCombination', 'PoorCare']:
        # For boolean variables, create a bar plot
        counts = df[var].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['False', 'True'])
    else:
        # For numeric variables, create a histogram
        ax.hist(df[var], bins=20)

    ax.set_title(labels[i], fontsize=12)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

# Remove the last two empty subplots
fig.delaxes(axes[-1])
fig.delaxes(axes[-2])

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
# plt.show()

# Randomly split the dataset into
# training set: 75% test
# testing set: 25%

df_train, df_test = train_test_split(df, test_size = 0.25, random_state = 42)

# Logistic Model
test_model = smf.logit("PoorCare ~ ERVisits + OfficeVisits + Narcotics + ProviderCount + NumberClaims + StartedOnCombination", data = df_train)

# Training
training_results = test_model.fit()

print(training_results.summary())

# Predicting probability of poor care for each patient in the training set
# Use 0.5 as a threshold to classify patients care
# Good Care (< 0.5) Bad Care (> 0.5)
df_train['PoorCare_pred'] = (training_results.predict() > 0.5).astype(int)

print(df_train.head())

pd.crosstab(df_train['PoorCare'], df_train['PoorCare_pred'])

# Calculating Overall Accuracy for Training Set
accuracy_train = accuracy_score(df_train['PoorCare'], df_train['PoorCare_pred'])

print("Accuracy on the training dataset:", accuracy_train)

# Baseline Accuracy
accuracy_train = accuracy_score(df_train['PoorCare'], np.zeros(len(df_train)))

print("Accuracy on the training dataset:", accuracy_train)

# Predictions on Test Set
df_test['PoorCare_pred'] = (training_results.predict(df_test) > 0.5).astype(int)
cm_t = pd.crosstab(df_test['PoorCare'], df_test['PoorCare_pred'])

accuracy_test = accuracy_score(df_test['PoorCare'], df_test['PoorCare_pred'])

print('Accuracy on the testing dataset:', accuracy_test)

# Extreme Threshold Low 0.1
df_train['PoorCare_pred_01'] = (training_results.predict() > 0.1).astype(int)
cm_pt1 = pd.crosstab(df_train['PoorCare'], df_train['PoorCare_pred_01'])

print(cm_pt1)

acc_train01 = accuracy_score(df_train['PoorCare'], df_train['PoorCare_pred_01'])

print('Accuracy for 0.1 Threshold:', acc_train01)

# Extreme Threshold High

df_train['PoorCare_pred_09'] = (training_results.predict() > 0.9).astype(int)
cm_pt9 = pd.crosstab(df_train['PoorCare'], df_train['PoorCare_pred_09'])

print(cm_pt9)

acc_train09 = accuracy_score(df_train['PoorCare'], df_train['PoorCare_pred_09'])

print('Accuracy for 0.9 threshold', acc_train09)

# Accuracy, Precision, and Recall for Training

# Calculate the accuracy on the training dataset
accuracy_train = accuracy_score(df_train['PoorCare'], df_train['PoorCare_pred'])
print("Accuracy on the training dataset:", accuracy_train)

# Calculate the precision on the training dataset
precision_train = precision_score(df_train['PoorCare'], df_train['PoorCare_pred'])
print("Precision on the training dataset:", precision_train)

# Calculate the recall on the training dataset
recall_train = recall_score(df_train['PoorCare'], df_train['PoorCare_pred'])
print("Recall on the training dataset:", recall_train)

# Accuracy, Precision, and Recall for Testing

# Calculate the accuracy on the testing dataset
accuracy_test = accuracy_score(df_test['PoorCare'], df_test['PoorCare_pred'])
print("Accuracy on the testing dataset:", accuracy_test)

# Calculate the precision on the testing dataset
precision_test = precision_score(df_test['PoorCare'], df_test['PoorCare_pred'])
print("Precision on the testing dataset:", precision_test)

# Calculate the recall on the testing dataset
recall_test = recall_score(df_test['PoorCare'], df_test['PoorCare_pred'])
print("Recall on the testing dataset:", recall_test)

# AUC Scores

# Making predictions on the training dataset
y_pred_train = training_results.predict(df_train)
print("Training Predictions: ", y_pred_train)

# Making predictions on the testing dataset
y_pred_test = training_results.predict(df_test)
print("Testing Predictions: ", y_pred_test)

# Calculating the AUC on the training dataset
auc_train = roc_auc_score(df_train['PoorCare'], y_pred_train)
print("AUC on the training dataset: ", auc_train)

# Calculating the AUC on the testing dataset
auc_test = roc_auc_score(df_test['PoorCare'], y_pred_test)
print("AUC on the testing dataset: ", auc_test)

# fpr - false positive rates
# tpr - true positive rates
# thresholds

fpr, tpr, thresholds = roc_curve(df_train['PoorCare'], y_pred_train)

# Plot the ROC curve

plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, label = f'AUC = {auc_train:.2f}')
plt.plot([0, 1], [0,1], 'k--', label = 'Random Chance') # Diagonal line represents random chance

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc = 'lower right')
plt.show()
