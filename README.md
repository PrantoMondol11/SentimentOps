# 🚀 End-to-End MLOps Pipeline with MLflow, DVC, AWS, Docker, Kubernetes, Prometheus & Grafana

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)]()
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green)]()
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue)]()
[![AWS](https://img.shields.io/badge/AWS-Cloud-yellow)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-blue)]()

---

## 📌 Overview

This project demonstrates a complete **production-grade MLOps workflow** for building, tracking, deploying, and monitoring a Machine Learning application.

The pipeline automates the entire machine learning lifecycle, from data ingestion to deployment and monitoring, while following industry-standard MLOps practices.

### Key Highlights

- 🔄 Automated ML Pipeline
- 📊 Experiment Tracking with MLflow
- 📦 Data Versioning using DVC
- ☁️ AWS Cloud Deployment
- 🐳 Docker Containerization
- ☸️ Kubernetes (EKS) Deployment
- 🚀 CI/CD using GitHub Actions
- 📈 Monitoring with Prometheus & Grafana

---

## 🎯 Business Objective

The goal of this project is to create a scalable and reproducible machine learning system that:

- Tracks experiments efficiently
- Versions datasets and models
- Automates deployment workflows
- Supports cloud-native infrastructure
- Enables continuous monitoring

---

## 🏗️ System Architecture

```text
Data Source
    │
    ▼
Data Ingestion
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
MLflow Model Registry
    │
    ▼
Flask Application
    │
    ▼
Docker Container
    │
    ▼
AWS ECR
    │
    ▼
AWS EKS Cluster
    │
    ▼
Prometheus
    │
    ▼
Grafana Dashboard
```

---

## 🛠️ Tech Stack

### Machine Learning

- Python 3.10
- Scikit-Learn
- Pandas
- NumPy

### Experiment Tracking

- MLflow
- DagsHub

### Data Versioning

- DVC
- AWS S3

### Backend

- Flask

### DevOps & MLOps

- Docker
- GitHub Actions
- AWS ECR
- AWS EKS
- Kubernetes

### Monitoring

- Prometheus
- Grafana

---

## 📂 Project Structure

```text
├── data/
├── notebooks/
├── src/
│   ├── logger/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── register_model.py
│
├── flask_app/
│
├── tests/
├── scripts/
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Features

### 📊 Experiment Tracking

- MLflow integration
- Parameter logging
- Metrics tracking
- Artifact management
- Model versioning

### 📦 Data Versioning

- DVC pipeline management
- Dataset tracking
- S3 remote storage integration

### 🚀 CI/CD Pipeline

GitHub Actions automates:

- Code validation
- Unit testing
- Docker image build
- Push image to AWS ECR
- Kubernetes deployment

### 🐳 Containerization

```bash
docker build -t capstone-app:latest .
docker run -p 8888:5000 capstone-app:latest
```

### ☸️ Kubernetes Deployment

Application deployed using:

- AWS Elastic Kubernetes Service (EKS)
- LoadBalancer Services
- Auto Scaling Infrastructure

---

## ☁️ AWS Services Used

| Service | Purpose |
|----------|----------|
| IAM | Access Management |
| S3 | Artifact & Dataset Storage |
| ECR | Docker Image Registry |
| EKS | Kubernetes Cluster |
| EC2 | Monitoring Servers |
| CloudFormation | Infrastructure Provisioning |

---

## 🔄 MLOps Workflow

```text
Developer Pushes Code
           │
           ▼
     GitHub Actions
           │
           ▼
       Run Tests
           │
           ▼
   Build Docker Image
           │
           ▼
      Push to ECR
           │
           ▼
      Deploy to EKS
           │
           ▼
      Monitor System
           │
           ▼
Prometheus + Grafana
```

---

## 📈 Monitoring & Observability

### Prometheus

Collects:

- Application Metrics
- Container Metrics
- Kubernetes Metrics
- Infrastructure Metrics

### Grafana

Visualizes:

- Request Traffic
- CPU Usage
- Memory Consumption
- Pod Health
- Service Availability

---

## 🚀 Installation

### Clone Repository

```bash
git clone [git repo link]
```

### Create Virtual Environment

```bash
conda create -n atlas python=3.10
conda activate atlas
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run DVC Pipeline

```bash
dvc repro
```

Check Pipeline Status:

```bash
dvc status
```

---

## ▶️ Run Flask Application

```bash
cd flask_app

python app.py
```

---

## 🐳 Docker Deployment

Build Image:

```bash
docker build -t sentimentops:latest .
```

Run Container:

```bash
docker run -p 8888:5000 sentimentops:latest
```

---

## ☸️ Kubernetes Deployment

Deploy Application:

```bash
kubectl apply -f deployment.yaml
```

Verify Deployment:

```bash
kubectl get pods
kubectl get svc
```

---

## 📊 Results

✅ Reproducible ML Pipeline

✅ Automated CI/CD Workflow

✅ Scalable Cloud Deployment

✅ Experiment Tracking & Model Registry

✅ End-to-End Monitoring & Observability

---

## 🔮 Future Improvements

- Model Drift Detection
- Automated Retraining
- Canary Deployments
- Multi-Environment Deployment
- Advanced Monitoring Alerts

---

## 👨‍💻 Author

### Pranto Mondol

**Machine Learning Engineer | MLOps Enthusiast | Cloud Practitioner**

- GitHub: https://github.com/mondolpranto83
- Email: mondolpranto83@gmail.com

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.
