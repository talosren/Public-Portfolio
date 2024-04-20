import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

df=pd.read_csv('Quality.csv')

df_train, df_test = train_test_split(df, test_size = 0.25, random_state = 42)

# define logistic model with training dataset
# define logistic model with training dataset
# model=smf.logit("PoorCare ~ ERVisits + Narcotics + StartedOnCombination", data=df_train) #remove non-significant variables one by one

model = smf.logit("PoorCare ~ OfficeVisits + Narcotics + StartedOnCombination", data = df_train)

# OfficeVisits + Narcotics + StartedOnCombination

# F-1 score:  0.6153846153846154

# Training Dataset Metrics:
#   Accuracy:  0.83
#   Precision: 0.79
#   Recall:    0.44
#   AUC:       0.83

# Threshold - 0.5
# Testing Dataset Metrics:
#   Accuracy:  0.85
#   Precision: 0.80
#   Recall:    0.50
#   AUC:       0.68

# Threshold - 0.3
# Testing Dataset Metrics:
#   Accuracy:  0.85
#   Precision: 0.71
#   Recall:    0.62
#   AUC:       0.68


# PoorCare_pred   0  1
# PoorCare
# 0              24  1
# 1               4  4

# 0              23  2
# 1               3  5

# train the model
res = model.fit()

# show the results
print(res.summary())

y_pred_train = res.predict(df_train)
y_pred_test = res.predict(df_test)

#select threshold
t = 0.3

df_train['PoorCare_pred'] = (res.predict(df_train) > t ).astype(int)

df_test['PoorCare_pred'] = (res.predict(df_test) > t ).astype(int)

# Calculate the accuracy on the training dataset
accuracy_train = accuracy_score(df_train['PoorCare'], df_train['PoorCare_pred'])

# Calculate the precision on the training dataset
precision_train = precision_score(df_train['PoorCare'], df_train['PoorCare_pred'])

# Calculate the recall on the training dataset
recall_train = recall_score(df_train['PoorCare'], df_train['PoorCare_pred'])

# Calculate the AUC on the training dataset
auc_train = roc_auc_score(df_train['PoorCare'], y_pred_train)

# Calculate the accuracy on the testing dataset
accuracy_test = accuracy_score(df_test['PoorCare'], df_test['PoorCare_pred'])

# Calculate the precision on the testing dataset
precision_test = precision_score(df_test['PoorCare'], df_test['PoorCare_pred'])

# Calculate the recall on the testing dataset
recall_test = recall_score(df_test['PoorCare'], df_test['PoorCare_pred'])

# Calculate the AUC on the testing dataset
auc_test = roc_auc_score(df_test['PoorCare'], y_pred_test)

cm_t=pd.crosstab(df_test['PoorCare'], df_test['PoorCare_pred'])
print(cm_t)

print("F-1 score: ", 2 * ((precision_test * recall_test)/(precision_test + recall_test)))

print("Threshold is:", t, "\n")
print("Training Dataset Metrics:")
print("  Accuracy:  {:.2f}".format(accuracy_train))
print("  Precision: {:.2f}".format(precision_train))
print("  Recall:    {:.2f}".format(recall_train))
print("  AUC:       {:.2f}".format(auc_train))
print()
print("Testing Dataset Metrics:")
print("  Accuracy:  {:.2f}".format(accuracy_test))
print("  Precision: {:.2f}".format(precision_test))
print("  Recall:    {:.2f}".format(recall_test))
print("  AUC:       {:.2f}".format(auc_test))

# Calculating the AUC on the training dataset
auc_train = roc_auc_score(df_train['PoorCare'], y_pred_train)
print("AUC on the training dataset: ", auc_train)

# Calculating the AUC on the testing dataset
auc_test = roc_auc_score(df_test['PoorCare'], y_pred_test)
print("AUC on the testing dataset: ", auc_test)

fpr, tpr, thresholds = roc_curve(df_train['PoorCare'], y_pred_train)

# Calculate Youden's J statistic for each threshold
j_scores = tpr - fpr

# Find the threshold that maximizes Youden's J statistic
optimal_threshold_idx_j = np.argmax(j_scores)
optimal_threshold_j = thresholds[optimal_threshold_idx]

# Alternatively, calculate the distance to the top-left corner of the ROC curve
distances_to_top_left = np.sqrt((1 - fpr)**2 + tpr**2)
optimal_threshold_idx_dist = np.argmin(distances_to_top_left)
optimal_threshold_dist = thresholds[optimal_threshold_idx]

# # Calculate the midpoint threshold
# midpoint_threshold = (optimal_threshold_j + optimal_threshold_dist) / 2

# Plot the ROC curve
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, label = f'AUC = {auc_train:.2f}')
plt.plot([0, 1], [0,1], 'k--', label = 'Random Chance') # Diagonal line represents random chance
plt.scatter(fpr[optimal_threshold_idx_j], tpr[optimal_threshold_idx_j], marker='o', color='g', label=f'Youden\'s J Optimal Threshold ({optimal_threshold_j:.2f})')
plt.scatter(fpr[optimal_threshold_idx_dist], tpr[optimal_threshold_idx_dist], marker='o', color='m', label=f'Distance Optimal Threshold ({optimal_threshold_dist:.2f})')
# plt.scatter(fpr[np.argmin(np.abs(thresholds - midpoint_threshold))], tpr[np.argmin(np.abs(thresholds - midpoint_threshold))], marker='o', color='b', label=f'Midpoint Threshold ({midpoint_threshold:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc = 'lower right')
plt.show()

# Optimal Threshold
# 0.3 obtains the strongest J-statistic trading precision for recall