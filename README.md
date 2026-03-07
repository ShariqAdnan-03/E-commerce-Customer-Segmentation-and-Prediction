# 🛒 E-Commerce Customer Market Segmentation

> Unsupervised customer segmentation using RFM Analysis, K-Means Clustering, and a Random Forest classifier — built on a real-world UK e-commerce transaction dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [RFM Analysis](#rfm-analysis)
- [Pre-Processing for Clustering](#pre-processing-for-clustering)
- [K-Means Clustering](#k-means-clustering)
- [Customer Segments](#customer-segments)
- [Classification Model](#classification-model)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds an end-to-end **customer segmentation system** for a UK-based e-commerce retailer. Using transactional data, each customer is profiled using **RFM (Recency, Frequency, Monetary)** scores and grouped into meaningful segments via **K-Means Clustering**. A **Random Forest classifier** is then trained on the labelled segments, enabling real-time prediction of which segment any new customer belongs to.

The final system powers personalised marketing strategies — moving the business from one-size-fits-all campaigns to targeted, data-driven engagement.

---

## Problem Statement

A business that treats every customer the same:
- Wastes marketing budget on customers who will never convert
- Fails to protect its highest-value buyers from churn
- Misses re-engagement opportunities for customers quietly drifting away

**Goal:** Identify distinct customer groups based on purchasing behaviour and build a scalable model to classify new customers automatically.

---

## Dataset

The dataset is a real-world transactional log from a UK-based online retailer.

| Feature | Description |
|---|---|
| `InvoiceNo` | Transaction ID. Prefix `C` = cancellation |
| `StockCode` | Unique product code |
| `Description` | Product name |
| `Quantity` | Units purchased. Negative = returned items |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit (£) |
| `CustomerID` | Unique customer identifier. Nulls = guest checkouts |
| `Country` | Customer's country |

> ⚠️ The raw dataset file is excluded from this repository due to size. Download it from the [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) and place it in the root directory as `data.csv`.

---

## Project Pipeline

```
Raw Data
   │
   ▼
Data Cleaning         → Handle nulls, capture returns, standardise formats
   │
   ▼
Feature Engineering   → Total_Price, temporal features, ReturnCount
   │
   ▼
EDA                   → Univariate, Bivariate, Multivariate analysis
   │
   ▼
RFM Table             → Recency, Frequency, Monetary per customer
   │
   ▼
Pre-Processing        → VIP isolation → Log transform → Standard scaling
   │
   ▼
K-Means Clustering    → Optimal K via Elbow + Silhouette → K = 3
   │
   ▼
Segment Labelling     → 4-tier business segments assigned
   │
   ▼
Classification        → Random Forest trained on labelled RFM data
   │
   ▼
Deployment            → Flask app for real-time segment prediction
```

---

## Exploratory Data Analysis

EDA is structured across three levels of analysis:

**Univariate Analysis — The "What & Where"**
- Top 10 best-selling products by transaction frequency
- Top 10 markets by transaction volume — UK accounts for ~96.6% of all transactions

**Bivariate Analysis — The "When"**
- Monthly revenue trend — confirms a holiday gifting spike in November–December
- Hourly traffic distribution — peak shopping window identified at **12 PM**

**Multivariate Analysis — The "Golden Window"**
- Day × Hour activity heatmap — reveals the exact day-time combinations for maximum campaign impact

---

## RFM Analysis

Each customer is collapsed from transaction rows into a single behavioural profile:

| Metric | Definition | Formula |
|---|---|---|
| **Recency** | Days since last purchase | `Snapshot Date − Last Invoice Date` |
| **Frequency** | Number of unique orders | `COUNT(DISTINCT InvoiceNo)` |
| **Monetary** | Total revenue generated | `SUM(Quantity × UnitPrice)` |

**RFM EDA Findings:**
- All three metrics show extreme right-skewness — a small number of wholesale buyers spend £50,000+
- Frequency and Monetary are strongly correlated (+0.55)
- Recency is negatively correlated with both — recent buyers also buy more and spend more

---

## Pre-Processing for Clustering

K-Means is a distance-based algorithm — it is highly sensitive to outliers and feature scale. Three preprocessing steps are applied before modelling:

**Step 1 — VIP Isolation**
Customers at or above the **99th monetary percentile (≥ £18,714)** are manually labelled as *VIP Segment* and excluded from the K-Means pool. This prevents extreme whale spend from collapsing all other customers into a single indistinct cluster.

**Step 2 — Log Transformation**
`log(x + 1)` applied to Recency, Frequency, and Monetary to compress extreme right-skewness and bring the distribution closer to normal.

**Step 3 — Standard Scaling**
Z-score normalisation applied so that Days, Order Counts, and Pounds all carry equal weight in the distance calculation. Output: mean = 0, std = 1 across all features.

---

## K-Means Clustering

To eliminate human bias in choosing the number of clusters, K was evaluated over the range **K = 2 to 10** using two independent metrics:

| Method | What it measures | Optimal K |
|---|---|---|
| **Elbow Curve (Inertia)** | Point of diminishing returns in within-cluster variance | K = 3 |
| **Silhouette Score** | How well-separated and compact clusters are (closer to +1 = better) | K = 3 |

Both methods independently confirmed **K = 3** as optimal for the core customer base.

---

## Customer Segments

The 3 K-Means clusters are profiled by their average RFM values and merged with the pre-labelled VIPs to produce the final **4-tier segmentation model**:

| Segment | Share | Recency | Frequency | Monetary | Business Action |
|---|---|---|---|---|---|
| **VIP Segment** | 9.9% | Low | Very High | ≥ £18,714 | Dedicated account managers, exclusive rewards, early product access |
| **Loyal Regulars** | 31.5% | Low | High | Moderate | Loyalty programme, referral bonuses, first-look newsletters |
| **Casual Buyers** | 20.3% | Medium | Low | Low | Personalised recommendations, abandoned cart flows, upsell campaigns |
| **At-Risk / Churning** | 38.2% | High | Declining | Low | Win-back emails, time-limited discounts, exit surveys |

---

## Classification Model

With segments labelled, a supervised classifier is trained to predict the segment of any new customer using only their RFM scores. Seven algorithms were benchmarked using `GridSearchCV` with cross-validation and hyperparameter tuning inside a `Pipeline`:

| Model | Accuracy | Notes |
|---|---|---|
| **Random Forest** | **~97%** | ✅ Champion model |
| LightGBM | ~95% | Strong runner-up |
| Decision Tree | ~93% | Interpretable, prone to overfitting |
| SVM | ~89% | Strong boundary, slower training |
| KNN | ~86% | Sensitive to scale |
| Logistic Regression | ~82% | Too linear for this problem |
| Naive Bayes | ~76% | Independence assumption violated by RFM correlations |

### Feature Importance (Random Forest)

| Feature | Importance | Interpretation |
|---|---|---|
| `Monetary` | 52% | Spend is the strongest predictor of segment |
| `Frequency` | 31% | Order count separates loyal from casual |
| `Recency` | 17% | Flags at-risk customers who have gone quiet |

> Monetary being the #1 driver validates the decision to manually isolate VIPs by monetary threshold before clustering.

---

## Results

- ✅ **4,335 customers** profiled and segmented from raw transactional data
- ✅ **44 VIP customers** isolated — spending 9× the customer average
- ✅ **K = 3** validated scientifically via dual-method evaluation
- ✅ **Random Forest** achieved ~97% classification accuracy on held-out test data
- ✅ **Monetary value** confirmed as the primary driver of segment membership (52%)
- ✅ **Flask app** deployed for real-time customer segment prediction

---

## Project Structure

```
customer-segmentation/
│
├── Cust_segment.ipynb              # Full analysis notebook (EDA → Modelling)
├── app.py                          # Flask web application
├── main.py                         # Entry point
├── description1.txt                # Feature/column descriptions
│
├── rfm_data_before_clustering.csv  # RFM table (pre-clustering)
│
├── customer_model.pkl              # Trained Random Forest model
├── model_features.pkl              # Feature names used during training
├── segment_encoder.pkl             # LabelEncoder for segment names
│
├── .gitignore                      # Excludes large data files and pkl artifacts
└── README.md                       # Project documentation
```

> 📁 `data.csv` and `final_segmented_ecommerce_data.csv` are excluded via `.gitignore` due to file size.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm flask
```

**3. Add the dataset**

Download `data.csv` from the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) and place it in the root folder.

**4. Run the notebook**
```bash
jupyter notebook Cust_segment.ipynb
```

**5. Launch the Flask app**
```bash
python app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-informational)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualisation-9cf)

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data wrangling and numerical computation |
| `matplotlib` / `seaborn` | Data visualisation |
| `scikit-learn` | Preprocessing, K-Means, classification models, GridSearchCV |
| `lightgbm` | Gradient boosting classifier |
| `flask` | Web application for real-time prediction |

---

<p align="center">
  <i>Built with 🔍 curiosity and ☕ coffee — E-Commerce Customer Segmentation Project</i>
</p>
