import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load your cleaned dataset
data = pd.read_csv('Crude.csv')

# Handle 'Consumer_Preferences' column
mode_preference = data['Consumer_Preferences'].mode().iloc[0]
data['Consumer_Preferences'] = data['Consumer_Preferences'].replace('oil', mode_preference)

# One-Hot Encode 'Consumer_Preferences'
data = pd.get_dummies(data, columns=['Consumer_Preferences'], prefix='Preference', drop_first=True)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data.drop(columns=['Crude_Oil_Demand_1000bpd']))
data_imputed = pd.DataFrame(X_imputed, columns=data.drop(columns=['Crude_Oil_Demand_1000bpd']).columns)

# 3. Train-Test Split
X = data_imputed
y = data['Crude_Oil_Demand_1000bpd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Selection and Training
rf = RandomForestRegressor()

# 5. Model Training and Evaluation with Cross-Validation
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 6. Automatic Leading Indicator Identification using RFE
# You can change the number of features to select based on your requirements
num_features_to_select = 5
rfe = RFE(estimator=best_rf, n_features_to_select=num_features_to_select)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]

print("Selected Leading Indicators:")
print(selected_features)

# 7. Predict Leading Indicators for Years 2023 to 2027
years = list(range(2023, 2028))  # Generate years from 2023 to 2027
X_future = pd.DataFrame({'Year': years})  # Create a DataFrame with the generated years

# Predict leading indicators for the next 5 years (replace with your own prediction method)
for feature in selected_features:
    # Assuming 'PredictLeadingIndicator' is your function to predict indicators
    X_future[f'Predicted_{feature}'] = PredictLeadingIndicator(X, feature)

# 8. Predict Global Crude Oil Demand for the Next 5 Years
# Combine historical values and predicted leading indicators
X_combined = pd.concat([X, X_future[selected_features]], axis=1)

# Predict global crude oil demand for the next 5 years
y_future = best_rf.predict(X_combined)

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame({'Year': years, 'Predicted_Crude_Oil_Demand_1000bpd': y_future})

# 9. Visualization of Predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Crude_Oil_Demand_1000bpd'], label='Actual Data', marker='o')
plt.plot(predictions_df['Year'], predictions_df['Predicted_Crude_Oil_Demand_1000bpd'], label='Predictions', marker='o')
plt.xlabel('Year')
plt.ylabel('Crude Oil Demand (1000bpd)')
plt.title('Crude Oil Demand Prediction')
plt.legend()
plt.grid(True)
plt.show()

# 10. Save predictions to a CSV file
predictions_df.to_csv('Crude_Oil_Demand_Predictions.csv', index=False)
