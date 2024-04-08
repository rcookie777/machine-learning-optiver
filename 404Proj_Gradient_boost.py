import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# load the dataset
df = pd.read_csv("/mnt/scratch/tairaeli/cse_dat/train.csv")

# drop na values
df = df.dropna()

# calulate ratio
df['bid_ask_ratio'] = df['bid_size']/df['ask_size']

# create target variable
df["target_change"] = np.ones_like(df["target"])
df.loc[df["target"]<0,'target_change'] = 0

# Select features and target
selected_features = ['ask_size', 'imbalance_size', 'imbalance_buy_sell_flag', 'matched_size', 'bid_size', 'bid_ask_ratio', 'reference_price']

# use only stock_id 0
df = df[df['stock_id'] == 0]
df_selected = df[selected_features]
target = df['target_change']

X = df_selected.values
y = target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def create_xgb_model(objective = 'binary:logistic', n_estimators=100, 
#                      learning_rate=0.1, max_depth=3,
#                      tree_method='auto'):
    
#     xgb_class = xgb.XGBClassifier(
#         objective=objective,
#         n_estimators=n_estimators,
#         learning_rate=learning_rate,
#         max_depth=max_depth,
#         tree_method=tree_method,
#         random_state=42)
    
#     return xgb_class

# param_grid = {
#     'objective': ['binary:logistic','binary:logitraw'],
#     'n_estimators': [10,20,50,100,200,500,1000],
#     'learning_rate': [0.5, 0.1, 0.01, 0.001, 0.0001],
#     'max_depth': [2,3,5,10],
#     'tree_method': ['exact','approx','hist']
# }

# print("Running Gridsearch xgb")

# grid = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, cv=3, n_jobs = 6)
# grid_result = grid.fit(X_train, y_train)

# print("Gridsearch complete!")

# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# testing best model found
xgb_class_best = xgb.XGBClassifier(learning_rate = 0.1,
                                   max_depth = 10,
                                   n_estimators = 500,
                                   tree_method = 'exact',
                                   objective = 'binary:logistic')

xgb_class_best.fit(X_train,y_train)

# evaluating accuracy of model
accuracy = xgb_class_best.score(X_test, y_test)
print("Accuracy:",accuracy)

y_pred = xgb_class_best.predict(X_test)

# generating confusion matrix
xgb_conf_mat = confusion_matrix(y_test, y_pred, normalize="all")
print(xgb_conf_mat)

# f1 score
print(f1_score(y_test,y_pred))

feature_importances = xgb_class_best.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(df_selected.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel('XGBoost Feature Importance')
plt.title('Feature Importance for Predicting Bid Size')
plt.savefig("XGB_Feature_Importance.png")
plt.show()
