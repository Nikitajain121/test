import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit app
st.title('Iris Flower Classifier')

# Input fields for user
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.4)
sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.4)
petal_length = st.slider('Petal Length', 1.0, 7.0, 1.3)
petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

# Make prediction
if st.button('Predict'):
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = rf_model.predict(user_input)
    st.write(f'The predicted Iris species is: {iris.target_names[prediction[0]]}')