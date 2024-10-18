# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset with correct path for Windows
credit_data = pd.read_csv(r'C:\Users\mjibr\Downloads\CreditDataset.csv')

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in credit_data.columns:
    if credit_data[column].dtype == 'object':  # Only encode categorical columns
        le = LabelEncoder()
        credit_data[column] = le.fit_transform(credit_data[column])
        label_encoders[column] = le

# Separate features (X) and target (y)
X = credit_data.drop(columns='Class')
y = credit_data['Class']

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for k-NN (it's recommended for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
dt_model = DecisionTreeClassifier(random_state=42)  # Decision Tree
knn_model = KNeighborsClassifier(n_neighbors=5)  # k-NN with k=5

# Train both models
dt_model.fit(X_train, y_train)  # Train Decision Tree on the original data
knn_model.fit(X_train_scaled, y_train)  # Train k-NN on the scaled data

# Make predictions with both models
y_pred_dt = dt_model.predict(X_test)  # Predictions for Decision Tree
y_pred_knn = knn_model.predict(X_test_scaled)  # Predictions for k-NN

# Evaluate the models using common metrics

# 1. Accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)  # Accuracy for Decision Tree
accuracy_knn = accuracy_score(y_test, y_pred_knn)  # Accuracy for k-NN

# 2. Precision
precision_dt = precision_score(y_test, y_pred_dt, average='macro')  # Precision for Decision Tree
precision_knn = precision_score(y_test, y_pred_knn, average='macro')  # Precision for k-NN

# 3. Recall
recall_dt = recall_score(y_test, y_pred_dt, average='macro')  # Recall for Decision Tree
recall_knn = recall_score(y_test, y_pred_knn, average='macro')  # Recall for k-NN

# 4. F1 Score
f1_dt = f1_score(y_test, y_pred_dt, average='macro')  # F1 score for Decision Tree
f1_knn = f1_score(y_test, y_pred_knn, average='macro')  # F1 score for k-NN

# 5. AUC and ROC for Decision Tree
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)  # AUC score

# Plot ROC Curve for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt, pos_label=1)
plt.figure()
plt.plot(fpr_dt, tpr_dt, color='blue', label='Decision Tree (AUC = {:.2f})'.format(roc_auc_dt))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend()
plt.show()

# Store all results in a dictionary
results = {
    'Accuracy': {'Decision Tree': accuracy_dt, 'k-NN': accuracy_knn},
    'Precision': {'Decision Tree': precision_dt, 'k-NN': precision_knn},
    'Recall': {'Decision Tree': recall_dt, 'k-NN': recall_knn},
    'F1 Score': {'Decision Tree': f1_dt, 'k-NN': f1_knn},
    'AUC': {'Decision Tree': roc_auc_dt}
}

# Display the results as a DataFrame
results_df = pd.DataFrame(results)
print(results_df)  # Replacing ace_tools with print to display results in terminal
