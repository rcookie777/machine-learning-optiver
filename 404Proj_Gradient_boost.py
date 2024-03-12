import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

df = df.dropna()

df['bid_ask_ratio'] = df['bid_size']/df['ask_size']
features_df = df[['ask_size', 'imbalance_size', 'matched_size', 'bid_size', 'bid_ask_ratio']]
target_df = df['reference_price']

X = features_df.values
y = target_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    tree_method='auto',
    random_state=42
)

xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Reference Price')
plt.ylabel('Predicted Reference Price')
plt.title('Actual vs. Predicted Reference Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.show()

feature_importances = xg_reg.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(features_df.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel('XGBoost Feature Importance')
plt.title('Feature Importance for Predicting Bid Size')
plt.show()
