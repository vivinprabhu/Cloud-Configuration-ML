import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('./cloud.csv')

df = df.rename(columns={
    'id': 'id',
    'timestamp': 'timestamp',
    'operatingSystem': 'os',
    'numberOfInstances': 'num_instances',
    'instanceName': 'instance_name',
    'vcpu': 'vcpu',
    'cpuUsage': 'cpu_usage',
    'memory': 'memory',
    'memoryUsage': 'memory_usage',
    'networkPerformance': 'network_performance',
    'storageType': 'storage_type',
    'numberOfVolume': 'number_of_volume',
    'storageSize': 'storage_size',
    'numberOfALB': 'ALB',
    'costPerMonth': 'cost_per_month'
})

X = df.drop(columns=['cost_per_month', 'network_performance', 'memory_usage', 'cpu_usage', 'timestamp'])
y = df['cost_per_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

categorical_cols = ['os', 'instance_name']
numeric_cols = ['num_instances', 'number_of_volume', 'storage_size', 'ALB', 'vcpu', 'memory']

# XGBoost

preprocessor_xgb = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_cols)
    ]
)

pipe_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor_xgb),
    ('clf', XGBRegressor(random_state=8))
])

search_space_xgb = {
    'clf__max_depth': Integer(3, 12),
    'clf__learning_rate': Real(0.001, 0.1, prior='log-uniform'),
    'clf__n_estimators': Integer(200, 2500),
    'clf__subsample': Real(0.6, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode': Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0),
    'clf__min_child_weight': Integer(1, 10),
    'clf__scale_pos_weight': Real(0.5, 1.0)
}

start_xgb = time.time()
opt_xgb = BayesSearchCV(pipe_xgb, search_space_xgb, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt_xgb.fit(X_train, y_train)
end_xgb = time.time()
y_pred_xgboost = opt_xgb.predict(X_test)
time_xgb = end_xgb - start_xgb


# SVM

preprocessor_svm = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_cols)
    ]
)

pipe_svm = Pipeline(steps=[
    ('preprocessor', preprocessor_svm),
    ('clf', SVR())
])

search_space_svm = {
    'clf__C': Real(1e-3, 1e3, prior='log-uniform'),
    'clf__epsilon': Real(0.01, 0.1),
    'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

start_svm = time.time()
opt_svm = BayesSearchCV(pipe_svm, search_space_svm, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt_svm.fit(X_train, y_train)
end_svm = time.time()
y_pred_svm = opt_svm.predict(X_test)
time_svm = end_svm - start_svm

# Random Forest model

pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_svm),
    ('clf', RandomForestRegressor(random_state=8))
])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)



# R2 Score

r2_xgboost = r2_score(y_test, y_pred_xgboost)
print(f'R-square of XGBoost: {r2_xgboost}')
r2_svm = r2_score(y_test, y_pred_svm)
print(f'R-square of SVM: {r2_svm}')
r2_rf = r2_score(y_test, y_pred_rf)
print(f'R-square of Random Forest: {r2_rf}')


# Mean Squared Error

mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
print(f'XGBoost MSE: {mse_xgboost}')
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f'SVM MSE: {mse_svm}')
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest MSE: {mse_rf}')


# Actual vs Predictions cost for all models

plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_svm, color='green', label='SVM Predictions', alpha=0.6)
plt.scatter(y_test, y_pred_rf, color='orange', label='Random Forest Predictions', alpha=0.6)
plt.scatter(y_test, y_pred_xgboost, color='blue', label='XGBoost Predictions', alpha=0.6)
plt.plot(y_test, y_test, color='black', linestyle='--', label='Actual Values')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions from XGBoost, SVM, and Random Forest')
plt.legend()
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())
plt.grid()
plt.tight_layout()
plt.show()


# Graph: Iterations vs Execution Time for XGBoost and SVM
plt.figure(figsize=(12, 8))

# Execution Time Comparison
# Graph: Execution Time for XGBoost and SVM
plt.figure(figsize=(10, 6))
plt.bar(['XGBoost', 'SVM'], [time_xgb, time_svm], color=['blue', 'green'])
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison: XGBoost vs SVM')
plt.tight_layout()
plt.show()