# 💎 Diamond Price Analysis
### Data Mining Final Project | CRISP-DM Methodology

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## Project Overview

A complete end-to-end data mining study on the **Kaggle Diamonds Dataset**,
following the industry-standard **CRISP-DM methodology**.

The project analyzes **53,772 real diamonds** to answer key business questions:
- What drives diamond prices?
- Are there natural market segments?
- Which diamonds are unusually priced?
- Can we predict cut quality automatically?

---

## Team

 Name 

| Mustafa Nabil | 
| Rawda Attia |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) |
| Original Size | 53,940 diamonds |
| After Cleaning | 53,772 diamonds |
| Features | 10 (carat, cut, color, clarity, depth, table, price, x, y, z) |
| Target Variable | price (USD) |

---

## Mining Techniques Applied

| Category | Technique | Purpose |
|----------|-----------|---------|
| Clustering | K-Means (k=4) | Group diamonds into market segments |
| Regression | Polynomial + Ridge | Predict diamond price (R² = 0.9858) |
| Anomaly Detection | One-Class SVM + PCA | Detect unusually priced diamonds |
| Classification | Random Forest (500 trees) | Predict cut grade (Accuracy = 78.11%) |

---
## Project Structure

    dm-diamonds-analysis/
    │
    ├── data/
    │   ├── diamonds.csv               ← Raw dataset from Kaggle
    │   ├── eda_plots.png              ← EDA visualizations
    │   ├── correlation.png            ← Correlation heatmap
    │   ├── boxplots.png               ← Box plots by category
    │   ├── data_quality.png           ← Data quality analysis
    │   ├── clustering_optimal_k.png   ← Elbow + Silhouette plots
    │   ├── clustering_results.png     ← Final clustering results
    │   ├── regression_results.png     ← Regression model results
    │   ├── anomaly_detection.png      ← Anomaly detection results
    │   └── classification_results.png ← Classification results
    │
    ├── notebooks/
    │   └── diamonds_analysis.ipynb    ← Full analysis notebook
    │
    ├── dashboard/
    │   └── app.py                     ← Streamlit dashboard
    │
    ├── requirements.txt               ← Python dependencies
    └── README.md                      ← This file

---

## Setup Instructions

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Mustafa-elsherif/dm-diamonds-analysis.git
cd dm-diamonds-analysis
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Verify Dataset
Make sure `diamonds.csv` is inside the `data/` folder.
If missing, download from [Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds)
and place it at `data/diamonds.csv`.

### Step 4 — Run the Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard opens automatically at `http://localhost:8501`

### Step 5 — Run the Notebook
Open VS Code → open `notebooks/diamonds_analysis.ipynb`
→ select Python (Anaconda) kernel → Run All Cells

---

## Dashboard Features

| Tab | Content |
|-----|---------|
| Overview | Dataset summary, distributions, feature guide |
| Clustering | 4 market segments with interactive charts |
| Price Predictor | Enter any diamond details → get estimated price |
| Anomaly Detection | 538 unusual diamonds with PCA visualization |
| Cut Predictor | Predict cut grade from physical measurements |
| Business Insights | Top 3 findings and full model performance |

**Global Filters:** Cut quality · Price range · Carat range

---

## Key Results

| Technique | Algorithm | Metric | Result | Goal |
|-----------|-----------|--------|--------|------|
| Clustering | K-Means k=4 | Silhouette Score | 0.2182 | 4 segments ✅ |
| Regression | Polynomial + Ridge | R² | 0.9858 | R² > 0.90 ✅ |
| Anomaly Detection | One-Class SVM + PCA | Anomalies found | 538 (1%) | Detected ✅ |
| Classification | Random Forest 500 trees | Accuracy | 78.11% | > 85% ❌ |

---

## Key Business Findings

**Finding 1 — Carat is King**
Carat weight has a 0.92 correlation with price.
A 1-carat diamond costs roughly 5× more than a 0.5-carat diamond.

**Finding 2 — Four Market Segments**
| Segment | Avg Price | Count | Market Share |
|---------|-----------|-------|--------------|
| Budget | $1,090 | 22,698 | 42% |
| Mid-range | $3,317 | 6,565 | 12% |
| Upper Mid-range | $4,671 | 16,362 | 30% |
| Luxury | $10,855 | 8,147 | 15% |

**Finding 3 — 538 Diamonds Need Review**
1% of diamonds are anomalously priced.
Could be pricing errors, data issues, or rare gems worth highlighting.

---

## CRISP-DM Compliance

| Phase | Activities | Status |
|-------|-----------|--------|
| Business Understanding | Define problem, goals, success criteria | ✅ |
| Data Understanding | EDA, distributions, correlations, outliers | ✅ |
| Data Preparation | Cleaning, encoding, scaling, feature engineering | ✅ |
| Modelling | 4 techniques applied and tuned | ✅ |
| Evaluation | Metrics evaluated against business goals | ✅ |
| Deployment | Interactive Streamlit dashboard | ✅ |

---

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
streamlit
jupyter
---

*Diamond Price Analysis · Data Mining Final Project · Python & Streamlit*

