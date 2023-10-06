import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load your cleaned dataset
data = pd.read_csv('Crude.csv')

# Split data into training (till 2020) and prediction (2023 to 2025) sets
train_data = data[data['Year'].between(2010, 2022)]
predict_data = data[data['Year'].between(2020, 2025)]  # Updated for 2020 to 2025

# Handle missing values using SimpleImputer for input features
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(train_data.drop(columns=['Crude_Oil_Demand_1000bpd']))
y_train = train_data['Crude_Oil_Demand_1000bpd']

# 4. Model Selection and Training
rf = RandomForestRegressor()

# 5. Model Training
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_imputed, y_train)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train_imputed, y_train)

# 6. Prepare Future Years Data for Predictions (2023 to 2025)
future_years = list(range(2020, 2026))  # Define the future years for prediction
X_future = pd.DataFrame({'Year': future_years})

# Ensure that your input features for the future years are complete and do not contain missing values
# For example, if you have columns 'Feature1' and 'Feature2', make sure they have values for future years.

# 7. Predict Crude Oil Demand for Future Years (2023 to 2025)
X_future_imputed = imputer.transform(predict_data.drop(columns=['Crude_Oil_Demand_1000bpd']))
future_predictions = best_rf.predict(X_future_imputed)

# 9. Visualization of Predictions
predictions_df = pd.DataFrame({'Year': future_years, 'Predicted_Crude_Oil_Demand_1000bpd': future_predictions})

# Create a new DataFrame for the Excel export with Year, Actual, and Predicted columns
export_df = pd.DataFrame({'Year': future_years})
export_df['Actual'] = predict_data['Crude_Oil_Demand_1000bpd'].values
export_df['Predicted'] = future_predictions

plt.figure(figsize=(10, 6))
plt.plot(export_df['Year'], export_df['Predicted'], label='Predicted', marker='o')
plt.plot(export_df['Year'], export_df['Actual'], label='Actual', marker='x')
plt.xlabel('Year')
plt.ylabel('Crude Oil Demand (1000bpd)')
plt.title('Crude Oil Demand Prediction (2020 to 2025)')
plt.legend()
plt.grid(True)

# 10. Save predictions to a CSV file and export data to Excel
export_df.to_csv('Crude_Oil_Demand_Predictions.csv', index=False)

# Export data to Excel with Year, Actual, and Predicted columns in the same sheet
export_df.to_excel('Crude_Oil_Demand_Predictions.xlsx', index=False)

plt.show()
