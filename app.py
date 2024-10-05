from flask import Flask, render_template, request
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skopt import BayesSearchCV
from skopt.space import Real, Integer

app = Flask(__name__)

# Load and preprocess the dataset
cloud_data = pd.read_csv('./cloud.csv')
cloud_data = cloud_data.rename(columns={
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

cloud_data = cloud_data.drop(columns=['memory_usage', 'storage_type', 'cpu_usage'])

df = cloud_data
X = df.drop(columns=['cost_per_month'])
y = df['cost_per_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

categorical_cols = ['os', 'instance_name']
numeric_cols = ['num_instances', 'vcpu', 'memory', 'number_of_volume', 'storage_size', 'ALB']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', XGBRegressor(random_state=8))
])

search_space = {
    'clf__max_depth': Integer(3, 12),
    'clf__learning_rate': Real(0.001, 0.1, prior='log-uniform'),
    'clf__n_estimators': Integer(200, 1500),
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

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='neg_mean_squared_error', random_state=8)
opt.fit(X_train, y_train)

best_model = opt.best_estimator_

combined_features = (
    cloud_data['num_instances'].astype(str) + ' ' +
    cloud_data['instance_name'] + ' ' +
    cloud_data['storage_size'].astype(str) + ' ' +
    cloud_data['cost_per_month'].astype(str)
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    predicted_cost = None
    error_message = None
    if request.method == 'POST':
        try:
            user_input = {
                'instance_name': request.form['instance_name'],
                'os': request.form['os'],
                'num_instances': int(request.form['num_instances']),
                'number_of_volume': int(request.form['number_of_volume']),
                'storage_size': float(request.form['storage_size']),
                'ALB': int(request.form['ALB']),
                'vcpu': int(request.form['vcpu']),
                'memory': int(request.form['memory'])
            }

            input_df = pd.DataFrame([user_input])
            input_df = input_df[['instance_name', 'os', 'num_instances', 'number_of_volume', 'storage_size', 'ALB', 'vcpu', 'memory']] 

            # Make prediction
            predicted_cost = best_model.predict(input_df)[0]
        except ValueError as e:
            error_message = f"Error: {str(e)}. Please ensure all fields are correctly filled out."

    return render_template('forecast.html', predicted_cost=predicted_cost, error_message=error_message)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    if request.method == 'POST':
        try:
            cur_num_instances = request.form['numberOfInstances']
            cur_instance_name = request.form['instanceName']
            cur_storage_size = request.form['storageSize']
            cur_cost_per_month = float(request.form['costPerMonth'])

            user_input = (
                cur_num_instances + ' ' +
                cur_instance_name + ' ' +
                cur_storage_size + ' ' +
                str(cur_cost_per_month)
            )

            user_input_vector = vectorizer.transform([user_input])
            user_similarity = cosine_similarity(user_input_vector, feature_vectors)
            similarity_score = list(enumerate(user_similarity[0]))
            sorted_similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            i = 1
            for instance in sorted_similar:
                index = instance[0]
                num_instances = cloud_data.loc[index, 'num_instances']
                instance_name = cloud_data.loc[index, 'instance_name']
                storage_size = cloud_data.loc[index, 'storage_size']
                cost_per_month = float(cloud_data.loc[index, 'cost_per_month'])

                if cost_per_month < cur_cost_per_month:
                    if i <= 10:
                        recommendations.append({
                            'Number of Instances': num_instances,
                            'Instance Name': instance_name,
                            'Storage Size': storage_size,
                            'Cost Per Month': cost_per_month
                        })
                        i += 1

            if not recommendations:
                recommendations.append({'message': 'Your current plan is already economically good.'})
        except ValueError:
            recommendations.append({'message': 'Invalid input. Please ensure all fields are correctly filled out.'})

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)