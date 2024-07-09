import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
PATH = 'sketch_results.csv'
data = pd.read_csv(PATH)

# Features and target variable
X = data[['n_rows', 'mean_abs_per_updates']]
y = data['n_duplicates']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
linear_model = LinearRegression()
gbr_model = GradientBoostingRegressor(random_state=42)

# Train and evaluate Linear Regression
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Train and evaluate Gradient Boosting Regressor
gbr_model.fit(X_train_scaled, y_train)
y_pred_gbr = gbr_model.predict(X_test_scaled)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Print the results
print(f"Linear Regression - MSE: {mse_linear}, R2: {r2_linear}")
print(f"Gradient Boosting Regressor - MSE: {mse_gbr}, R2: {r2_gbr}")
