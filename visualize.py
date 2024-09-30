import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
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

search_space = {
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

opt = BayesSearchCV(pipe_xgb, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt.fit(X_train, y_train)

y_pred_xgboost = opt.predict(X_test)

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

pipe_svm.fit(X_train, y_train)

y_pred_svm = pipe_svm.predict(X_test)

r2_xgboost = r2_score(y_test, y_pred_xgboost)
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)

r2_svm = r2_score(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)

plt.figure(figsize=(10, 10))

plt.scatter(y_test, y_pred_xgboost, color='blue', label='XGBoost Predictions', alpha=0.6)
plt.scatter(y_test, y_pred_svm, color='green', label='SVM Predictions', alpha=0.6)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions from XGBoost and SVM')
plt.legend()
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())

plt.tight_layout()
plt.show()

print("XGBoost R-squared: ", r2_xgboost)
print("XGBoost Mean Squared Error: ", mse_xgboost)
print("SVM R-squared: ", r2_svm)
print("SVM Mean Squared Error: ", mse_svm)