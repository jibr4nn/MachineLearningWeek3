# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv(r'C:\Users\mjibr\Downloads\MaterialStrength.csv')

# Preprocess the data
# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':  # Encode only categorical columns
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Separate features (X) and target (y)
X = data.drop(columns='target_feature')
y = data['target_feature']

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for k-NN (it's recommended for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
dt_regressor = DecisionTreeRegressor(random_state=42)  # Decision Tree Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # k-NN Regressor with k=5

# Train both models
dt_regressor.fit(X_train, y_train)  # Train Decision Tree on the original data
knn_regressor.fit(X_train_scaled, y_train)  # Train k-NN on the scaled data

# Make predictions with both models
y_pred_dt = dt_regressor.predict(X_test)  # Predictions for Decision Tree
y_pred_knn = knn_regressor.predict(X_test_scaled)  # Predictions for k-NN

# Evaluate the models using common regression metrics

# Root Mean Squared Error (RMSE)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))  # RMSE for Decision Tree
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))  # RMSE for k-NN

# Mean Squared Error (MSE)
mse_dt = mean_squared_error(y_test, y_pred_dt)  # MSE for Decision Tree
mse_knn = mean_squared_error(y_test, y_pred_knn)  # MSE for k-NN

# R-squared (Coefficient of Determination)
r2_dt = r2_score(y_test, y_pred_dt)  # R-squared for Decision Tree
r2_knn = r2_score(y_test, y_pred_knn)  # R-squared for k-NN

# Store all results in a dictionary for comparison
results = {
    'RMSE': {'Decision Tree': rmse_dt, 'k-NN': rmse_knn},
    'MSE': {'Decision Tree': mse_dt, 'k-NN': mse_knn},
    'R-squared': {'Decision Tree': r2_dt, 'k-NN': r2_knn}
}

# Display the results as a DataFrame
results_df = pd.DataFrame(results)
print(results_df)  # Print the results for comparison
