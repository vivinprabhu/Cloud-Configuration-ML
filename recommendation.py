import pandas as pd

cloud_data = pd.read_csv('./cloud.csv')


print("Number of rows and columns")
print(cloud_data.shape)

print("First 5 data")
print(cloud_data.head())

print("Last 5 data")
print(cloud_data.tail())


# selecting the relevant features for recommendation

selected_features = ['numberOfInstances','instanceName','numberOfVolume','storageSize','numberOfALB','costPerMonth']
print(selected_features)


for feature in selected_features:
  cloud_data[feature] = cloud_data[feature].fillna('')

# combining all the 6 selected features
combined_features = (
    cloud_data['numberOfInstances'].astype(str) + ' ' +
    cloud_data['instanceName'] + ' ' +
    cloud_data['numberOfVolume'].astype(str) + ' ' +
    cloud_data['storageSize'].astype(str) + ' ' +
    cloud_data['numberOfALB'].astype(str) + ' ' +
    cloud_data['costPerMonth'].astype(str) + ' ' 
    )

print(combined_features)


# converting the text data to feature vectors
# TF-IDF stands for Term Frequency-Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)



# getting the similarity scores using cosine similarity
# similarity between 2 vectors

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(feature_vectors)

print(similarity)
print("Number of rows and columns after cosine_similarity:")
print(similarity.shape)


cur_num_instances = input('Enter the number of instances : ')
cur_instance_name = input('Enter your currect instance name: ')
cur_num_vol = input('Enter number of storage volume: ')
cur_storage_size = input('Enter your current storage size: ')
cur_alb = input('Enter number of alb: ')
cur_cost_per_month = input('Enter your current cost (last month): ')

# Combine user inputs into a single string
user_input = (
    cur_num_instances + ' ' +
    cur_instance_name + ' ' +
    cur_num_vol + ' ' +
    cur_storage_size + ' ' +
    cur_alb+ ' ' +
    cur_cost_per_month
)

# Vectorize the user input
user_input_vector = vectorizer.transform([user_input])

# Compute similarity between the user input and the dataset
user_similarity = cosine_similarity(user_input_vector, feature_vectors)

# Get a list of similarity scores
similarity_score = list(enumerate(user_similarity[0]))
#print(similarity_score)

# Sort the recommendations based on similarity scores
sorted_similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#print(sorted_similar)

print('Recommendations for you with lower cost per month:\n')

i = 1
cur_cost_per_month = float(cur_cost_per_month)  

for instance in sorted_similar:
    index = instance[0]
    num_instances = cloud_data.loc[index, 'numberOfInstances']
    instance_name = cloud_data.loc[index, 'instanceName']
    number_of_volume = cloud_data.loc[index, 'numberOfVolume']
    storage_size = cloud_data.loc[index, 'storageSize']
    number_of_alb = cloud_data.loc[index, 'numberOfALB']
    cost_per_month = cloud_data.loc[index, 'costPerMonth']
    
    if float(cost_per_month) < cur_cost_per_month:
        if i <= 10:
            print(f"{i}. Number of Instances: {num_instances}, Instance Name: {instance_name}, Number of Volume : {number_of_volume} ,Storage Size: {storage_size}, Number of ALB : {number_of_alb} ,Cost Per Month: {cost_per_month}")
            i += 1

if i == 1:
    print("Your current plan is already economically good.")