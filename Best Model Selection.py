# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.impute import SimpleImputer  # Import the SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error


# Load your cleaned dataset
data = pd.read_csv('Crude.csv')


# - Ensure data is cleaned and prepared.

# 2. Feature Selection/Engineering
# - Identify relevant features and perform any necessary feature engineering.

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can change the strategy if needed
X_imputed = imputer.fit_transform(data.drop(columns=['Crude_Oil_Demand_1000bpd']))
data_imputed = pd.DataFrame(X_imputed, columns=data.drop(columns=['Crude_Oil_Demand_1000bpd']).columns)

# Rename columns to replace whitespace with underscores
data_imputed.columns = data_imputed.columns.str.replace(' ', '_')

# 3. Train-Test Split
X = data_imputed  # Now use the imputed and renamed data
y = data['Crude_Oil_Demand_1000bpd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Selection
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    # Removed 'Prophet' model from the dictionary
    'LightGBM': lgb.LGBMRegressor(
        verbose=-1,  # Suppress LightGBM warnings
        # Add other hyperparameters here
    )
}

# 5. Model Training and Evaluation
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results[model_name] = {
        'MSE': mse,
        'R-squared': r_squared,
        'RMSE': rmse,
        'MAE': mae
    }

# 6. Hyperparameter Tuning (for selected models)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lgb = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [20, 31, 40],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_child_samples': [10, 20, 30]
}

# Hyperparameter tuning for Random Forest
rf = RandomForestRegressor()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Hyperparameter tuning for LightGBM
lgbm = lgb.LGBMRegressor(
    verbose=-1,  # Suppress LightGBM warnings
    # Add other hyperparameters here
)
grid_search_lgb = GridSearchCV(lgbm, param_grid_lgb, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
grid_search_lgb.fit(X_train, y_train)

best_lgbm = grid_search_lgb.best_estimator_
y_pred_lgbm = best_lgbm.predict(X_test)
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)

# 7. Final Model Selection
best_model = min(results, key=lambda x: results[x]['MSE'])

# 8. Explain Why the Model is Chosen
print(f"The best model is '{best_model}' with the following metrics:")
print(f"MSE: {results[best_model]['MSE']}")
print(f"R-squared: {results[best_model]['R-squared']}")
print(f"RMSE: {results[best_model]['RMSE']}")
print(f"MAE: {results[best_model]['MAE']}")
print("This model is chosen because it has the lowest Mean Squared Error (MSE), "
      "indicating that it provides the best predictive performance on the test data.")

# 9. Predictions
# Use the selected best_model to make predictions on new data.

# 10. Ensemble Methods (if needed)
# You can implement ensemble methods to combine the strengths of multiple models.

# 11. Fine-tune the code based on your specific dataset and requirements.

