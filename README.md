# 🛒 Flipkart Reviews Sentiment Analysis

### Internship Project | GenAI

---

## 📌 Project Overview
This project was completed during my **GenAI Internship** as part of a practical ML task.

The objective of this project is to build and deploy a **sentiment analysis application** that predicts whether a Flipkart product review is **Positive or Negative**.


---

## ⚙️ How the Project Works

### 📊 Data Used
- Flipkart product reviews dataset
- Reviews are already labeled for sentiment

---

### 🔢 Feature Representation
- Used a preprocessed dataset
- Converted review text into numerical format using a saved vectorizer
- Ensured the same vectorizer is reused during inference to avoid mismatch

---

### 🧠 Model Training
- Trained a classification model on processed data
- Evaluated model performance using multiple metrics
- Serialized the trained model and vectorizer for reuse

---

### 🖥️ Application Development
- Developed an interactive web application using **Streamlit**
- Users can input a product review
- The application loads the trained model and returns sentiment predictions in real time

---

### 🚀 Deployment
- Deployed the application on **Streamlit Cloud**
- Used relative file paths for cloud compatibility
- Debugged file system and environment-related issues during deployment

---

## 🛠️ Tech Stack
- Python  
- Scikit-learn  
- Streamlit  
- Git & GitHub  

---

##  Future Improvements:
- No experiment tracking
- No CI/CD pipeline
- No automated retraining or monitoring

---

## 🔗 Live Application
👉 https://flipkartreviewssentimentanalysis-axohthgmbaksov3oofdy2p.streamlit.app/
