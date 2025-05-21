# 🌸 Iris Flower Classifier - Streamlit App

This project is a simple **machine learning web application** built using **Streamlit** that classifies Iris flowers into one of three species: **Setosa**, **Versicolor**, or **Virginica**. The app allows users to input flower measurements using sliders and returns the predicted species using a trained **Random Forest Classifier**.

---

## 📊 Dataset

The model uses the **Iris dataset** from `sklearn.datasets`. It contains 150 samples with 4 features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Each sample is labeled as one of the following classes:

- `0` - Setosa  
- `1` - Versicolor  
- `2` - Virginica

---

## 🔍 Features

- User-friendly UI with interactive sliders for input.
- Real-time prediction of Iris flower species.
- Uses **Random Forest** for classification (high accuracy and robustness).
- Built with **Streamlit** for fast prototyping.

---

## 🚀 How to Run the App

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/iris-streamlit-classifier.git
cd iris-streamlit-classifier
2. Install the dependencies
bash
Copy
Edit
pip install streamlit pandas scikit-learn
3. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
🧠 Model Overview
Model: RandomForestClassifier (n_estimators=100)

Training/Test Split: 80/20

Evaluation: Trained on default Iris dataset with good generalization


📁 File Structure
bash
Copy
Edit
iris-streamlit-classifier/
│
├── app.py              # Main Streamlit app
├── README.md           # Project documentation
├── requirements.txt  
