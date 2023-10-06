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
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

# Load your cleaned dataset
data = pd.read_csv('Crude.csv')

# Separate training data (years up to 2022)
train_data = data[data['Year'] <= 2022]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(train_data.drop(columns=['Crude_Oil_Demand_1000bpd']))
y_train = train_data['Crude_Oil_Demand_1000bpd']

# Model Selection and Training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(
        verbose=-1,  # Suppress LightGBM warnings
        # Add other hyperparameters here
    )
}

# Model Training and Evaluation
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_imputed, y_train)

    # Split training data for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_values = []

    for train_index, val_index in tscv.split(X_train_imputed):
        X_train_cv, X_val = X_train_imputed[train_index], X_train_imputed[val_index]
        y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate MSE for this fold
        mse_fold = mean_squared_error(y_val, y_pred)
        mse_values.append(mse_fold)

    # Calculate average MSE across folds
    avg_mse = np.mean(mse_values)

    results[model_name] = {
        'Avg. MSE': avg_mse
    }

# Final Model Selection
best_model = min(results, key=lambda x: results[x]['Avg. MSE'])

# Explain Why the Model is Chosen
print(f"The best model is '{best_model}' with the following metrics:")
print(f"Avg. MSE: {results[best_model]['Avg. MSE']}")
print("This model is chosen because it has the lowest average Mean Squared Error (MSE) "
      "across cross-validation folds, indicating better performance on the training data.")

# Fine-tune the code based on your specific dataset and requirements.
