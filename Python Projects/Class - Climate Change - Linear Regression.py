import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

# Import
df_climate = pd.read_csv('data/climate_change.csv')
col = []
for i in df_climate.columns:
    i = i.replace('-','_')
    i = i.replace('(','_')
    i = i.replace(')','_')
    col.append(i)
df_climate.columns = col

# 1 Data visualization
# Year - (x) - Temp (y)

min_year = df_climate['Year'].min()
max_year = df_climate['Year'].max()

sns.lineplot(data = df_climate, x = 'Year',
             y = 'Temp', color = 'skyblue').set_title('Temperature'
             + ' Changes Over Time')
plt.xlim(min_year, max_year)  # Set the x-axis range from the minimum to maximum year to make a full graph
# plt.grid()
# plt.show()

#3. Model training + Linear Regression

training_data = df_climate[df_climate['Year'] <= 2006]
testing_data = df_climate[df_climate['Year'] >= 2007]

training_model = smf.ols("Temp ~ MEI + CO2 + CH4 + N2O + "
                         "CFC_11 + CFC_12 + TSI + Aerosols", data = training_data)
result_training = training_model.fit()

# 4. R^2 Values

print(f'\n Training - 2006 BELOW \n')
print(result_training.summary())
# -------- Training ----------
# R-Square - 0.751 - 75.1%
# Adj. R-Square - 0.744 - 744%
# F-statistic - 103.6/

yhat = result_training.predict(testing_data)

# True value of dependent variable
y = testing_data['Temp']

ybar = np.mean(testing_data['Temp'])
SSE = np.sum((y - yhat)**2)
SST = np.sum((y - ybar)**2)
R2 =  1 - SSE/SST
# -------- Training ----------
# R-Square - 0.1837 - 18.37%

# 5. Statistical Signficance
# P-Value - Below 0.05
# MEI, CO2, CH4, CFC_11, CFC_12, TSI, Aerosols

# 6.Results interpretation
# N2O - Coef - (-0.0165) - Insignificant
# CFC-11 - Coef - (-0.0066) - Significant

# Higher concentrations of N2O and CFC-11 result in lower temperatures
# Counterintuitive as these are green houses gases which we want to lower

# 7. Correlation Table
corr = training_data[training_data.columns].corr()
sns.heatmap(data = corr, annot = True, cmap = 'coolwarm')
# plt.show()

# 8. High Correlations (0.7)
# N2O is highly correlated with CO2, CH4, and CFC_12
# CH4 is highly correlated with N2O, CFC_11, CFC_12, CO2
# CFC_12 is highly correlated with N2O, CO2, CH4, CFC_11
# CFC_11 is highly correlated with CH4

# Avoid multicollinearity
# Use significant variables only
# p-value less than 0.05

# 9. Building the best data
best_model = smf.ols("Temp ~ MEI + TSI + Aerosols +"
                  "CFC_12 + CFC_11 + C(Month)",
                  data = training_data)
result_best = best_model.fit()
print(result_best.summary())

# R^2 Value
# -------- BEST ----------
# R-Square - 0.753 - 75.3%
# Adj. R-Square - 0.748 - 74.8%
# F-statistic - 141.0

yhat = result_best.predict(testing_data)

# True value of temperature
y = testing_data['Temp']

ybar = np.mean(testing_data['Temp'])
SSE = np.sum((y - yhat)**2)
SST = np.sum((y - ybar)**2)
R2 =  1 - SSE/SST
# -------- Training ----------
# R-Square - 0.2183 - 21.83% - No Categorical Month
# R-Square - 0.2614 - 26.14% - Categorical Month <- Higher explanation
print(R2)

# -----------Conclusion----------------
# This means that independent variables:
# [MEI, TSI, Aerosols, CFC_12, CFC_11 + Month]
# explain 26.14% of the variance
# in the dependent variable (Temp)