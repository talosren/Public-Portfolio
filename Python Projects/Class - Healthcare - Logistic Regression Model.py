import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

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
        sns.countplot(x=var, data=df, ax=ax)
        ax.set_xticklabels(['False', 'True'])
    else:
        # For numeric variables, create a histogram
        sns.histplot(df[var], bins = 20, ax = ax)

    ax.set_title(labels[i], fontsize=12)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

# Remove the last two empty subplots
fig.delaxes(axes[-1])
fig.delaxes(axes[-2])

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()