# Heart Disease Prediction MLOps Pipeline

GitHub Repository Link:
https://github.com/2025cs05002Kavya/heart-disease-mlops

## Video Demonstration

Video walkthrough link:
https://drive.google.com/your-link

## Overview

This project implements an end-to-end MLOps pipeline for predicting heart disease using machine learning and modern DevOps practices.

The project includes:
- Data preprocessing and EDA
- Machine learning model training
- MLflow experiment tracking
- FastAPI model serving
- Docker containerization
- Kubernetes deployment
- GitHub Actions CI/CD
- Monitoring and logging

---

## Dataset

Heart Disease UCI Dataset

Source:
https://archive.ics.uci.edu/ml/datasets/heart+Disease

---

## Models Used

- Logistic Regression
- Random Forest Classifier

Random Forest performed better with higher accuracy and F1-score.

---

## Running Locally

### Install Dependencies

pip install -r requirements.txt

### Start API

uvicorn api.app:app --reload

---

## Docker

### Build Docker Image

docker build -t heart-api .

### Run Docker Container

docker run -p 8000:8000 heart-api

---

## Kubernetes Deployment

kubectl apply -f k8s/deployment.yaml

kubectl apply -f k8s/service.yaml

---

## CI/CD Pipeline

GitHub Actions pipeline automates:
- dependency installation
- unit testing
- model training

---

## Monitoring & Logging

The FastAPI application logs prediction results and confidence scores for monitoring purposes.

---

Local API Testing:

Run:
uvicorn api.app:app --reload

Open:
http://127.0.0.1:8000/docs

---

## Author

2025cs05002Kavya