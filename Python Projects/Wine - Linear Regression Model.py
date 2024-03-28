import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

wine_df = pd.read_csv('data/Wine.csv')
wineTest_df = pd.read_csv('data/WineTest.csv')

print(wine_df.head(3))
print(wine_df.columns)

# Independent Variables (X)
# Year, WinterRain, AGST, HarvestRain, Age, FrancePop

# Dependent Variables (Y)
# Price

# Looking at the different independent variables compared to the dependent variable (Wine Price)
# We're looking for correlations between them to see if these independent variables can predict (Wine Price)
# sns.scatterplot(data=wine_df, x='Year', y='Price')
# plt.title('Year vs Price')
# plt.tight_layout()
# plt.show()
#
# # Scatter plot for 'WinterRain' vs 'Price'
# sns.scatterplot(data=wine_df, x='WinterRain', y='Price')
# plt.title('WinterRain vs Price')
# plt.tight_layout()
# plt.show()
#
# # Scatter plot for 'AGST' vs 'Price'
# sns.scatterplot(data=wine_df, x='AGST', y='Price')
# plt.title('AGST vs Price')
# plt.tight_layout()
# plt.show()
#
# # Scatter plot for 'HarvestRain' vs 'Price'
# sns.scatterplot(data=wine_df, x='HarvestRain', y='Price')
# plt.title('HarvestRain vs Price')
# plt.tight_layout()
# plt.show()
#
# # Scatter plot for 'Age' vs 'Price'
# sns.scatterplot(data=wine_df, x='Age', y='Price')
# plt.title('Age vs Price')
# plt.tight_layout()
# plt.show()
#
# # Scatter plot for 'FrancePop' vs 'Price'
# sns.scatterplot(data=wine_df, x='FrancePop', y='Price')
# plt.title('FrancePop vs Price')
# plt.tight_layout()
# plt.show()

# Now let's Define the model for Linear Regression to determine
# Correlation between these variables and wine price

model_all = smf.ols("Price ~ AGST + WinterRain + HarvestRain + Age + FrancePop", data = wine_df)
result_all = model_all.fit()
print(result_all.summary())
# -------- ALL ----------
# R-Square - 0.829 - 82.9%
# Adj. R-Square - 0.784 - 78.4%
# F-statistic - 18.47

# Predictions + Errors
df_all = wine_df.copy()
df_all['yhat'] = result_all.predict()
df_all['error'] = df_all['Price'] - df_all['yhat']
df_all['residual'] = result_all.resid

print()
print(df_all.head(5))
print()

# However this isn't the best ^^
model_best = smf.ols("Price ~ AGST + WinterRain + HarvestRain + Age", data = wine_df)
result_best = model_best.fit()
print(result_best.summary())
# -------- BEST ----------
# R-Square - 0.829 - 82.9%
# Adj. R-Square - 0.794 - 79.4% # Higher Adj R-Square means that FrenchPop had 0 impact
# F-statistic - 24.17

# Predictions - WineTest
yhat = result_best.predict(wineTest_df)

# True price of wine
y = wineTest_df['Price']

# Average Price of Wine
ybar = np.mean(wine_df['Price'])
SSE = np.sum((y - yhat)**2) # 0.06926280848776642
SST = np.sum((y - ybar)**2) # 0.3369268563519997
R2 =  1 - SSE/SST # 0.79442776026318


# -----------Conclusion----------------
# This means that independent variables:
# [Year, WinterRain, AGST, HarvestRain, Age]
# explain 79.44% of the variance
# in the dependent variable
# We can effectively predict the price of wine through these variables!