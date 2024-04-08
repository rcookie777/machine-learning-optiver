import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')

# Drop missing values
df = df.dropna()

# Calculate bid-ask ratio as a feature
df['bid_ask_ratio'] = df['bid_size'] / df['ask_size']

# Create a binary target variable for price increase (1) or decrease (0)
# df['price_change'] = (df['wap'] > df['reference_price']).astype(int)
df["target_change"] = np.ones_like(df["target"])
df.loc[df["target"]<0,'target_change'] = 0

features_df = df[['bid_ask_ratio', 'imbalance_buy_sell_flag', 'reference_price', 'ask_size', 'bid_size', 'imbalance_size', 'matched_size']]
#features_df = df[['imbalance_buy_sell_flag', 'bid_ask_ratio', 'time_id', 'date_id']]
target_df = df['target_change']

X = features_df.values
y = target_df.values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
xg_class = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xg_class.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xg_class.predict(X_test)

# Calculating the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Feature importances
feature_importances = xg_class.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(features_df.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel('XGBoost Feature Importance')
plt.title('Feature Importance for Predicting Price Change')
plt.show()
