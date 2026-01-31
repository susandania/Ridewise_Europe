# RideWise – Customer Churn Prediction & Analytics Engine

## Overview

RideWise is a mobility technology analytics project focused on **customer churn prediction, segmentation, and deployment of a production-grade machine learning service** using a mid-level MLOps setup.

The project is structured to reflect how a real European mobility company would approach customer analytics, retention strategy, and operational ML deployment across multiple cities.

---

## Project Objectives

- Understand customer behaviour through EDA and segmentation
- Build a churn prediction model with strong business interpretability
- Apply feature engineering using behavioural and temporal signals
- Deploy a prediction API using FastAPI and Docker
- Practice production-aware ML workflows (CI/CD, monitoring, structure)

---

## Project Structure

```text
RideWise_Europe/
│
├── data/
│   ├── raw/                  # Source datasets (CSV)
│   ├── processed/            # Cleaned and feature-ready datasets
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_segmentation.ipynb # Customer segmentation
│   └── 03_churn_model.ipynb  # Churn model development
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py    # Data ingestion and validation
│   │   └── build_features.py # Feature engineering pipeline
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   ├── api/
│   │   └── app.py            # FastAPI prediction service
│   │
│   └── utils/
│       └── config.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/datascience-muhammad/RideWise_Europe.git
cd RideWise_Europe
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Access

The project uses structured customer, trip, promotion, and activity datasets stored in the `data/raw/` directory.

These datasets are treated as source data for analysis, feature engineering, and model development. Contributors should focus on **understanding, validating, and transforming the data**, rather than how it was originally created.

---

## Workflow Expectations (Interns)

- Work only on your feature branch
- Commit frequently with clear messages
- Keep notebooks clean and reproducible
- Do not commit raw CSV files to GitHub

---

## Success Criteria

By the end of the project, contributors should be able to:

- Explain the structure and meaning of the datasets
- Justify feature choices
- Train and evaluate a churn model
- Interpret results in business terms
- Demonstrate a working ML pipeline

---

## License & Usage

This project is intended to mirror **industry-style analytics and ML delivery workflows** and is used for training and capability development.

---

**Maintainer**: RideWise Project Team  
**Last Updated**: January 2026


# RideWise Europe - Docker

Build:
docker build -t ridewise-api .

Run:
docker run -p 8000:8000 ridewise-api

Health:
http://127.0.0.1:8000/health