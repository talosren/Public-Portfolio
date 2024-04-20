import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

data = pd.read_csv("data/covid_cleaned3.csv", usecols = lambda column: column not in ['Unnamed: 0'])

print('Shape of data: ', data.shape)
print(data.head(5))

# Distribution of deaths
ax = sns.countplot(x = 'DEATH', data = data, palette = sns.cubehelix_palette(2))
plt.bar_label(ax.containers[0])
plt.title('Death Distribution', fontsize = 18, color = 'red')

# Creating Datasets
death_1_sample = data[data['DEATH'] == 1].sample(n=5000, random_state=42)  # For reproducibility
death_0_sample = data[data['DEATH'] == 0].sample(n=5000, random_state=42)

# Combine the samples into one DataFrame
df = pd.concat([death_1_sample, death_0_sample])

# Number of unique values by columns
for i in df.columns:
    print(i, '=>\t', len(df[i].unique()))

plt.figure(figsize = (15, 10))
sns.heatmap(df.corr(), annot = True, fmt = '.2f')
plt.title('Correlation Between Features', fontsize = 18, color = 'red')
# plt.show()

x = df.drop(columns = 'DEATH')
y = df['DEATH']

# Training, Validation, Test Set
train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size = 0.4, random_state = 42)
# Using Temp X to separate data into validation and testing data
# 0.4 Allocated to Temp X
# 0.6 Allocated to training data

val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size = 0.5, random_state = 42)
# Validation: 0.5 of the temp (0.4 of the data of initial (x, y) data
# Test: 0.5 of the temp (0.4 of the data of initial (x, y) data

print("Train_x :", train_x.shape)
print("Val_x :", val_x.shape)
print("Test_x :", test_x.shape)
print("Train_y :", train_y.shape)
print("Val_y :", val_y.shape)
print("Test_y :", test_y.shape)

# Training
print(train_x.columns)

# Logistic Regression

# Model 1 ALL
train_data = train_x.copy()
train_data['DEATH'] = train_y # Making a copy of Y values (DEATH) into combined VALIDATION DATA

formula_1 = 'DEATH ~ ' + ' + '.join(train_x.columns) # No Death in here
model_1 = smf.logit(formula = formula_1, data = train_data)
res_1 = model_1.fit()
print(res_1.summary())

# Model 2 Significant Only (p > 0.1)
formula_2 = 'DEATH ~ SEX + HOSPITALIZED + PNEUMONIA + AGE + DIABETES + IMMUNOSUPPRESSION + OTHER_DISEASE + RENAL_CHRONIC + COVID_POSITIVE'
model_2 = smf.logit(formula = formula_2, data = train_data)
res_2 = model_2.fit()
print(res_2.summary()) # 0.6180 Pseudo R-Square

# Model 3 No OTHER_DISEASE
formula_3 = 'DEATH ~ SEX + HOSPITALIZED + PNEUMONIA + AGE + DIABETES + IMMUNOSUPPRESSION + RENAL_CHRONIC + COVID_POSITIVE'
model_3 = smf.logit(formula = formula_3, data = train_data)
res_3 = model_3.fit()
print(res_3.summary()) # 0.6176 Pseudo R-Square

# AUC comparison ONLY GIVEN PROBABILITIES
val_data = val_x.copy()
val_data['DEATH'] = val_y # Making a copy of Y values (DEATH) into combined VALIDATION DATA

# Predictions
y_pred_val_2 = res_2.predict(val_data) # Model 2
y_pred_val_3 = res_3.predict(val_data) # Model 3

# AUC
auc_val_2 = roc_auc_score(val_data['DEATH'], y_pred_val_2)
auc_val_3 = roc_auc_score(val_data['DEATH'], y_pred_val_3)

# Results
print("Model 2's AUC on the validation dataset:", auc_val_2)
print("Model 3's AUC on the validation dataset:", auc_val_3)
# Model 3 is slightly better by 0.0004

# Classification Tree
leaves = 50

dt = DecisionTreeClassifier(min_samples_leaf = leaves, max_depth = 4, random_state = 42)
dt.fit(train_x, train_y) # Using the training Data

# [:, 0] - Finding probabilities of times when data is 0
# [:, 1] - Finding probabilities of times when data is 1

val_pred = dt.predict(val_x) # Default Threshold of t = 0.5
val_proba = dt.predict_proba(val_x)[:, 1] # Positive Cases only AKA DEATH probabilities

# Metrics of validation set
accuracy = accuracy_score(val_y, val_pred) # Comparing val_y to val_pred from decision tree
auc = roc_auc_score(val_y, val_proba) # AUC - PROBABILITY
recall = recall_score(val_y, val_pred)
precision = precision_score(val_y, val_pred)
f1 = f1_score(val_y, val_pred)

# Print validation set evaluation metrics
print("Val Set Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"AUC: {auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

# Relationship with confusion matrix
cm = confusion_matrix(val_y, val_pred)

print('Confusion Matrix:')
print(cm)

# Row: Actual = 0, Actual = 1
# Column: Prediction = 0: Prediction = 1

plt.figure(figsize = (15, 10))
plot_tree(dt, feature_names = train_x.columns, class_names = ['LIVED', 'DIED'], filled = True)
# plt.show()

results = []

# Combination of samples and depth
for min_samples in [40, 50, 200, 100]:
    for depth in [4, 5, 6, 9]:
        dt = DecisionTreeClassifier(min_samples_leaf = min_samples, max_depth = depth, random_state = 42)
        dt.fit(train_x, train_y)
        val_pred = dt.predict(val_x)
        val_proba = dt.predict_proba(val_x)[:, 1]

        # Metrics
        accuracy = accuracy_score(val_y, val_pred)
        auc = roc_auc_score(val_y, val_proba)
        recall = recall_score(val_y, val_pred)
        precision = precision_score(val_y, val_pred)
        f1 = f1_score(val_y, val_pred)

        results.append((min_samples, depth, accuracy, auc, recall, precision, f1))

results_df = pd.DataFrame(results, columns = ['min_samples_leaf', 'max_depth', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1 Score'])

best_model = results_df.loc[results_df['AUC'].idxmax()]

print('Best model configuration: ')
print(best_model)

# Random Forest

# Hyperparameters
n_estimators = 100 # Number of decision trees to make
max_depth = 5 # Number of legs (how deep)

rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 42)
rf.fit(train_x, train_y)

val_pred_rf = rf.predict(val_x)
val_proba_rf = rf.predict_proba(val_x)[:, 1]

# Calculate metrics
accuracy_rf = accuracy_score(val_y, val_pred_rf)
auc_rf = roc_auc_score(val_y, val_proba_rf)
recall_rf = recall_score(val_y, val_pred_rf)
precision_rf = precision_score(val_y, val_pred_rf)
f1_rf = f1_score(val_y, val_pred_rf)

# Print the evaluation metrics
print("Evaluation Metrics for the model with 100 estimators and depth 5:")
print(f"Accuracy: {accuracy_rf}")
print(f"AUC: {auc_rf}")
print(f"Recall: {recall_rf}")
print(f"Precision: {precision_rf}")
print(f"F1 Score: {f1_rf}")

results_rf = []

# Testing different combinations of n_estimators and max_depth
for estimators in [25, 50, 100, 200]:
    for depth in [3, 5, 8, 10]:
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=42)
        rf.fit(train_x, train_y)
        val_pred_rf = rf.predict(val_x)
        val_proba_rf = rf.predict_proba(val_x)[:, 1]

        # Calculate metrics
        accuracy_rf = accuracy_score(val_y, val_pred_rf)
        auc_rf = roc_auc_score(val_y, val_proba_rf)
        recall_rf = recall_score(val_y, val_pred_rf)
        precision_rf = precision_score(val_y, val_pred_rf)
        f1_rf = f1_score(val_y, val_pred_rf)

        results_rf.append((estimators, depth, accuracy_rf, auc_rf, recall_rf, precision_rf, f1_rf))

# Convert results to a DataFrame
results_rf_df = pd.DataFrame(results_rf, columns = ['Estimators', 'Depth', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1 Score'])

best_model = results_rf_df.loc[results_rf_df['AUC'].idxmax()]

print('Best model configuration')
print(best_model)

# Retraining and Evaluating the best model
combined_x = pd.concat([train_x, val_x])
combined_y = pd.concat([train_y, val_y])

# Logistic Regression
combined_data = combined_x.copy()
combined_data['DEATH'] = combined_y

test_data = test_x.copy()
test_data['DEATH'] = test_y

formula_3 = 'DEATH ~ SEX + HOSPITALIZED + PNEUMONIA + AGE + DIABETES + IMMUNOSUPPRESSION + RENAL_CHRONIC + COVID_POSITIVE'

# Retrain the model with the combined dataset
model_3_combined = smf.logit(formula=formula_3, data=combined_data)
res_3_combined = model_3_combined.fit()

# Evaluate on the test set
test_proba_log = res_3_combined.predict(test_data) # probability
test_pred_log = (test_proba_log > 0.5).astype(int) # class with threshold = 0.5

# Calculate metrics for the test set
accuracy_test = accuracy_score(test_y, test_pred_log)
auc_test = roc_auc_score(test_y, test_proba_log) # PROB FOR AUC
recall_test = recall_score(test_y, test_pred_log)
precision_test = precision_score(test_y, test_pred_log)
f1_test = f1_score(test_y, test_pred_log)

# Metrics
print("Test Set Evaluation Metrics:")
print(f"Accuracy: {accuracy_test}")
print(f"AUC: {auc_test}")
print(f"Recall: {recall_test}")
print(f"Precision: {precision_test}")
print(f"F1 Score: {f1_test}")

# Decision Tree
# Found through best IDX on AUC
best_samples_leaf = 40
best_depth = 5

best_dt = DecisionTreeClassifier(min_samples_leaf = best_samples_leaf, max_depth = best_depth, random_state = 42)
best_dt.fit(combined_x, combined_y)

test_pred_dt = best_dt.predict(test_x)
test_proba_dt = best_dt.predict_proba(test_x)[:, 1]

# Step 5: Calculate metrics for the test set
accuracy_test = accuracy_score(test_y, test_pred_dt)
auc_test = roc_auc_score(test_y, test_proba_dt)
recall_test = recall_score(test_y, test_pred_dt)
precision_test = precision_score(test_y, test_pred_dt)
f1_test = f1_score(test_y, test_pred_dt)

# Step 6: Print test set evaluation metrics
print("Test Set Evaluation Metrics:")
print(f"Accuracy: {accuracy_test}")
print(f"AUC: {auc_test}")
print(f"Recall: {recall_test}")
print(f"Precision: {precision_test}")
print(f"F1 Score: {f1_test}")

# Random Forest
# Step 2: Using the best model settings found earlier
best_estimators = int(best_model['Estimators'])
best_depth = int(best_model['Depth'])

# Step 3: Retrain the model with the combined dataset
best_rf = RandomForestClassifier(n_estimators=best_estimators, max_depth=best_depth, random_state=42)
best_rf.fit(combined_x, combined_y)

# Step 4: Evaluate on the test set
test_pred_rf = best_rf.predict(test_x)
test_proba_rf = best_rf.predict_proba(test_x)[:, 1]

# Step 5: Calculate metrics for the test set
accuracy_test = accuracy_score(test_y, test_pred_rf)
auc_test = roc_auc_score(test_y, test_proba_rf)
recall_test = recall_score(test_y, test_pred_rf)
precision_test = precision_score(test_y, test_pred_rf)
f1_test = f1_score(test_y, test_pred_rf)

# Step 6: Print test set evaluation metrics
print("Test Set Evaluation Metrics:")
print(f"Accuracy: {accuracy_test}")
print(f"AUC: {auc_test}")
print(f"Recall: {recall_test}")
print(f"Precision: {precision_test}")
print(f"F1 Score: {f1_test}")