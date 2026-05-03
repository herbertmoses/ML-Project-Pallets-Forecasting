# 📦 ML Project – Pallets Forecasting

A multi-service machine learning web application for forecasting pallet allocation and dehire quantities using Flask-based microservices.

---

## 🚀 Project Overview

This project predicts pallet demand using trained machine learning models across four independent services:

* 🪵 Wooden Pallet Allocation Forecast
* 🪵 Wooden Pallet Dehire Forecast
* ♻️ Plastic Pallet Allocation Forecast
* ♻️ Plastic Pallet Dehire Forecast

Each service is built as an individual Flask application and can be run independently or together using a centralized runner.

---

## 🧠 Key Features

* Multi-service Flask architecture
* Machine Learning model integration (scikit-learn)
* SQLite-based model storage (for plastic dehire)
* Centralized execution via `main.py`
* GitHub Actions CI pipeline
* Lightweight UI for predictions
* Python-centric backend design

---

## 📁 Project Structure

```
ML-Project-Pallets-Forecasting/
│
├── main.py
├── requirements.txt
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Wodden_allot_forecast/
│   ├── app.py
│   ├── model1.pkl
│   ├── templates/
│   └── static/
│
├── Wodden_dehired_forecast/
│   ├── app.py
│   ├── model1.pkl
│   ├── templates/
│   └── static/
│
├── plastic_allot_forecast/
│   ├── app.py
│   ├── model1.pkl
│   ├── templates/
│   └── static/
│
├── plastic_dehired_forecast/
│   ├── app.py
│   ├── dehired_forecast.db
│   ├── templates/
│   └── static/
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/<your-username>/ML-Project-Pallets-Forecasting.git
cd ML-Project-Pallets-Forecasting
```

---

### 2. Create Virtual Environment

```
python -m venv env
source env/bin/activate   # Mac/Linux
env\Scripts\activate      # Windows
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Option 1: Run All Services (Recommended)

```
python main.py
```

### 🌐 Access Services

| Service         | URL                   |
| --------------- | --------------------- |
| Wooden Allot    | http://127.0.0.1:5000 |
| Wooden Dehired  | http://127.0.0.1:5001 |
| Plastic Allot   | http://127.0.0.1:5002 |
| Plastic Dehired | http://127.0.0.1:5003 |

---

### Option 2: Run Individual Service

```
cd Wodden_allot_forecast
python app.py
```

---

## 🧪 CI/CD Pipeline

GitHub Actions workflow (`ci.yml`) performs:

* Dependency installation
* Module import validation
* Application import checks
* Basic ML prediction smoke test

Triggered on:

* Push to `main`
* Pull requests to `main`

---

## 🧩 Tech Stack

* **Backend:** Python, Flask
* **ML Models:** scikit-learn
* **Database:** SQLite (for model storage)
* **Frontend:** HTML, CSS
* **CI/CD:** GitHub Actions

---

## ⚠️ Known Limitations

* Models were trained using different scikit-learn versions (warnings may appear)
* No authentication or user management
* UI is minimal and intended for demo purposes
* Each service runs independently (not yet unified)

---

## 🔮 Future Enhancements

* Merge into a single FastAPI gateway
* Dockerize services with docker-compose
* Add REST APIs instead of form-based UI
* Centralized logging and monitoring (Grafana)
* React-based dashboard UI
* Model retraining pipeline

---

## 🤝 Contribution

Feel free to fork this repository and submit pull requests for improvements.

---
