## Customer Churn Prediction using ANN:

This project is a web-based application for predicting customer churn using an Artificial Neural Network (ANN). Built using Streamlit, this app allows users to input customer details and get real-time predictions on whether a customer is likely to churn.

## Features:

Predicts churn based on customer information.

Clean, interactive user interface built with Streamlit.

Dynamic input form for customer attributes.

Visualizations including heatmaps and boxplots for data exploration.

Trained using TensorFlow ANN and preprocessed with scikit-learn.

Supports model persistence using .h5 and .pkl files.


## How to Run:

Clone the repository:

git clone https://github.com/yourusername/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

## Create a virtual environment (optional but recommended):

python -m venv venv

venv\Scripts\activate          # On Windows

## Install the required packages:

pip install -r requirements.txt

## Run the Streamlit app:

streamlit run app.py

## 📈 Model Overview:

Architecture: 3-layer ANN with ReLU activations and dropout regularization.

Training Data: Preprocessed Telco Customer Churn dataset.

Output: Binary classification (Churn / No Churn).

Performance: 79% accuracy on test data.

## 📊 Visualizations Included:

Correlation heatmap.

Boxplot of monthly charges by churn status.

More can be added via Streamlit as needed.

## Requirements:

See requirements.txt or install key dependencies:

pip install streamlit pandas scikit-learn tensorflow seaborn matplotlib joblib

## ✅ Future Improvements:

Add SHAP explainability for predictions.

Deploy to Streamlit Cloud or other hosting service.

Save prediction history to database or CSV.

Add model training interface to web app.

## License:

This project is licensed under the MIT License.

Copyright (c) 2025 SATHISH B

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## User Interface:
![image alt](https://github.com/SathishB-1/Customer-Churn-Prediction/blob/856d6ed05f4d9eef20403f9f8f4a7ce6f72881a9/image/interface_1.png)
![image alt](https://github.com/SathishB-1/Customer-Churn-Prediction/blob/eaa9e56a7751051f0f37d5076d1bf3c1ea928d64/image/interface_2.png)
