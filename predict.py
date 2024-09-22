import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.metrics import r2_score

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

X = df.drop(columns=['cost_per_month' , 'network_performance' ,  'memory_usage', 'cpu_usage' ])
y = df['cost_per_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

categorical_cols = ['os', 'instance_name']
numeric_cols = ['num_instances', 'number_of_volume', 'storage_size', 'ALB', 'vcpu', 'memory']

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols), #MinMax Normalization (Avg)
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()), #Z-score normalization (Standard deviation 0-1) #similar scales
        ]), numeric_cols)
    ]
)

# Pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', XGBRegressor(random_state=8))
])

# Bayesian optimization
search_space = {
    'clf__max_depth': Integer(3, 12),  # max depth of a tree
    'clf__learning_rate': Real(0.001, 0.1, prior='log-uniform'), 
    'clf__n_estimators': Integer(200, 2500),  # number of trees
    'clf__subsample': Real(0.6, 1.0), # if subsample is set to 0.8, then 80% of the training data is randomly selected and used to build each tree
    'clf__colsample_bytree': Real(0.5, 1.0), # fraction of features used by each tree
    'clf__colsample_bylevel': Real(0.5, 1.0), # fraction of features used at each level of the tree
    'clf__colsample_bynode': Real(0.5, 1.0), #fraction of features used for each split in the tree
    'clf__reg_alpha': Real(0.0, 10.0), # L1 regularization - ingnore certain features
    'clf__reg_lambda': Real(0.0, 10.0), # L2 regularization - prevent overfitting
    'clf__gamma': Real(0.0, 10.0), # Minimum loss reduction required to make a further partition on a leaf node
    
    'clf__min_child_weight': Integer(1, 10), # Minimum sum of instance weights needed in a child
    'clf__scale_pos_weight': Real(0.5, 1.0) # Balances the positive and negative weights
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt.fit(X_train, y_train)

y_pred = opt.predict(X_test)

print("R-squared: ", r2_score(y_test, y_pred)) 


def predict_from_input(user_input):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[categorical_cols + numeric_cols]
    prediction = opt.predict(input_df)
    return prediction[0]

def get_user_input():
    user_input = {}
    user_input['instance_name'] = input("Enter instance name (e.g., 'r5.4xlarge'): ")
    user_input['os'] = input("Enter os name (e.g., 'Linux'): ")
    user_input['num_instances'] = int(input("Enter number of instances (e.g., 5): "))
    user_input['number_of_volume'] = float(input("Enter number of Volume (e.g., 6): "))
    user_input['storage_size'] = float(input("Enter storage size in GB (e.g., 698): "))
    user_input['ALB'] = float(input("Enter number of ApplicationLoadBalancer (e.g., 1): "))
    user_input['vcpu'] = float(input("Enter number of vcpu (e.g., 8): "))
    user_input['memory'] = float(input("Enter number of memory (e.g., 64 , 3749.73): "))
    return user_input

user_input = get_user_input()

predicted_cost = predict_from_input(user_input)
print("Predicted Cost per Month:", predicted_cost)