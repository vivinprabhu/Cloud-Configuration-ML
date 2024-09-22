# Machine Learning Cost Prediction and Resource Recommendation

This project leverages machine learning techniques to predict costs and recommend cloud resources. The application uses XGBoost regression for cost prediction and a TF-IDF based similarity measure for resource recommendation. It is built with a web interface using Flask, HTML, and CSS.

## Project Overview

- **Cost Prediction:** Uses XGBoost regression to forecast the cost of cloud resources based on various features.
- **Resource Recommendation:** Provides recommendations for cloud resources based on similarity to user inputs.

## Technologies Used

- **Machine Learning:** XGBoost, Bayesian Optimization
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Data Processing:** pandas, scikit-learn
- **Text Vectorization:** TfidfVectorizer

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/vivinprabhu/Cloud-Configuration-ML.git
    ```

2. **Download the dataset:**

   Ensure that the dataset `cloud.csv` is present in the project directory. The dataset should have the following columns: `numberOfInstances`, `instanceName`, `storageSize` and others required for the model.

## Usage

1. **Run the Flask application:**

    ```bash
    python app.py
    ```

2. **Access the web application:**

   Open your web browser and go to `http://127.0.0.1:5000/` to interact with the application.

   - **Home Page:** Displays the home interface.
   - **Recommendation Page:** Provides recommendations based on user inputs.
   - **Forecast Page:** Allows users to input features to forecast cost.

## Project Structure

- `app.py`: The main Flask application file that includes routes and logic.
- `cloud.csv`: Dataset used for training and prediction (ensure this file is present).
- `templates/`: Contains HTML files for the web interface.
- `static/`: Contains CSS files and other static assets.

## Requirements

Ensure you have the following Python packages installed:

- Flask
- pandas
- scikit-learn
- xgboost
- category_encoders
- numpy
- scipy

You need to install dependencies some are mentioned (install all other necessaries):

```bash
pip install flask pandas scikit-learn xgboost category_encoders numpy scipy 
