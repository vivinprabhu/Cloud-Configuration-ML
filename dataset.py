import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Constants for instance data
instances = {
    't3.medium': {'vcpu': 2, 'memory': 4, 'networkPerformance': 'Up to 5 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 21.39},
    't3.xlarge': {'vcpu': 4, 'memory': 16, 'networkPerformance': 'Up to 5 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 85.70},
    'c5.2xlarge': {'vcpu': 4, 'memory': 16, 'networkPerformance': 'Up to 10 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 173.74},
    'm5.2xlarge': {'vcpu': 8, 'memory': 32, 'networkPerformance': 'Up to 10 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 200.02},
    'r5.2xlarge': {'vcpu': 4, 'memory': 32, 'networkPerformance': 'Up to 10 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 265.72},
    'r5.4xlarge': {'vcpu': 8, 'memory': 64, 'networkPerformance': 'Up to 10 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 532.17},
    'c5.12xlarge': {'vcpu': 16, 'memory': 128, 'networkPerformance': '12 Gigabit', 'storageType': 'EBS only', 'costPerMonth': 1043.90}
}

# Generate random data
data = []
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

for i in range(20000):
    # Generate random timestamp between start_date and end_date
    timestamp = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
    
    # Operating System
    operating_system = "Linux"
    
    # Random number of instances
    number_of_instances = random.randint(1, 10)
    
    # Random instance name
    instance_name = random.choice(list(instances.keys()))
    
    # Fetching the instance details
    vcpu = instances[instance_name]['vcpu']
    memory = instances[instance_name]['memory']
    network_performance = instances[instance_name]['networkPerformance']
    storage_type = instances[instance_name]['storageType']
    cost_per_month = instances[instance_name]['costPerMonth'] * number_of_instances
    
    # Generating usage metrics
    cpu_usage = round(random.uniform(1, 100), 6)
    memory_usage = round(random.uniform(1, 100), 6)
    
    # Number of volumes
    number_of_volume = random.randint(1, 10)
    
    # Storage size between 1 and 1000
    storage_size = random.randint(1, 1000)
    
    # Number of ALB
    number_of_alb = random.randint(1, 5)
    alb_cost = number_of_alb * 43.80
    
    # Final cost per month including storage and ALB
    final_cost = round(cost_per_month + (number_of_volume * storage_size * 0.26) ,4)
    
    # Append data
    data.append([i+1, timestamp, operating_system, number_of_instances, instance_name, vcpu, cpu_usage, memory, memory_usage,
                 network_performance, storage_type, number_of_volume, storage_size, number_of_alb, final_cost])

# Create DataFrame
columns = ["id", "timestamp", "operatingSystem", "numberOfInstances", "instanceName", "vcpu", "cpuUsage", "memory", "memoryUsage", 
           "networkPerformance", "storageType", "numberOfVolume", "storageSize", "numberOfALB", "costPerMonth"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("cloud.csv", index=False)

# Display the first few rows of the dataset
print(df.head())