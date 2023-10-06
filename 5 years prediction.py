import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import xgboost as xgb

# Load your cleaned dataset
data = pd.read_csv('Crude.csv')

# Split data into training (till 2020) and prediction (2023 to 2025) sets
train_data = data[data['Year'].between(2010, 2022)]
predict_data = data[data['Year'].between(2020, 2025)]

# Handle missing values using SimpleImputer for input features
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(train_data.drop(columns=['Crude_Oil_Demand_1000bpd']))
y_train = train_data['Crude_Oil_Demand_1000bpd']

# Model Selection and Training (Using XGBoost)
xgb_reg = xgb.XGBRegressor()

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_imputed, y_train)

best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train_imputed, y_train)

# Feature Importance Analysis
feature_importances = best_xgb.feature_importances_

# Create a DataFrame to store feature importances and their corresponding names
feature_names = train_data.drop(columns=['Crude_Oil_Demand_1000bpd']).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Prepare Future Years Data for Predictions (2023 to 2025)
future_years = list(range(2020, 2026))
X_future = pd.DataFrame({'Year': future_years})

# Ensure that your input features for the future years are complete and do not contain missing values

# Predict Crude Oil Demand for Future Years (2023 to 2025)
X_future_imputed = imputer.transform(predict_data.drop(columns=['Crude_Oil_Demand_1000bpd']))
future_predictions = best_xgb.predict(X_future_imputed)

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot 1: Feature Importance
ax1 = axes[0]
ax1.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#6DFDD2')
ax1.set_xlabel('Feature Importance')
ax1.set_ylabel('Feature')
ax1.set_title('Feature Importance Analysis')
ax1.invert_yaxis()  # Invert the y-axis to show the most important features at the top
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# Create a DataFrame for the Excel export with Year, Actual, and Predicted columns
export_df = pd.DataFrame({'Year': future_years})
export_df['Actual'] = predict_data['Crude_Oil_Demand_1000bpd'].values
export_df['Predicted'] = future_predictions

# Plot 2: Crude Oil Demand Predictions
ax2 = axes[1]
ax2.plot(export_df['Year'], export_df['Predicted'], label='Predicted', marker='o', linestyle='--', color='#142459')
ax2.plot(export_df['Year'], export_df['Actual'], label='Actual', marker='x', color='#1AC9E6')
ax2.set_xlabel('Year')
ax2.set_ylabel('Crude Oil Demand (1000bpd)')
ax2.set_title('Crude Oil Demand Prediction (2020 to 2025)')
ax2.legend()
ax2.grid(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Save predictions to a CSV file and export data to Excel
export_df.to_csv('Crude_Oil_Demand_Predictions.csv', index=False)

# Export data to Excel with Year, Actual, and Predicted columns in the same sheet
export_df.to_excel('Crude_Oil_Demand_Predictions.xlsx', index=False)

# Display feature importance
print(feature_importance_df)
