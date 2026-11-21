# AI-Based Customer Segmentation and Intelligent Marketing Recommendation System

An enterprise-grade Data Science and Machine Learning platform that converts raw e-commerce transaction logs into behavioral customer segments (RFM + K-Means) and prescribes automated, targeted marketing strategies.

---

## 🚀 Key Highlights
- **Real-Data Pipeline:** Tested and validated on the full UCI Supermarket transaction corpus (541,910 rows).
- **Automated Data Sanitization:** Handles guest checkouts, cancellations (`InvoiceNo` with 'C'), negative rates, and duplicates.
- **RFM Feature Engineering:** Computes Recency, Frequency, Monetary values with quantile scoring (1–5) and variance-stabilizing log transformation.
- **Unsupervised Machine Learning:** Lloyd's K-Means clustering with K-Means++ initialization, Elbow Method evaluation (WCSS), Silhouette Score analysis, and 2D PCA dimensionality reduction.
- **Intelligent Marketing Engine:** Translates mathematical cluster centroids into commercial segments (Champions, Loyalists, Potential Loyalists, At-Risk, and Lost Customers) with actionable marketing playbooks.
- **Dual SaaS Dashboards:** Specialized interfaces for **Data Analysts** (pipeline tuning & ML diagnostics) and **Sellers** (business KPIs, customer search & export).

---

## 🛠️ Technology Stack
- **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js 4.4
- **Backend:** Python 3.10+, Flask
- **Data Science & ML:** Pandas, NumPy, SciPy (Spatial & Clustering), Scikit-learn
- **Database:** SQLite (built-in relational persistence with secure password hashing)

---

## 📁 Project Folder Structure
```
customer-segmentation-system/
├── app.py                      # Main Flask application and route controllers
├── config.py                   # Configuration parameters and storage paths
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── data/
│   ├── raw/                    # Raw transaction datasets (supermarket_data.csv)
│   └── processed/              # Cleaned CSVs, RFM metrics, and segmented outputs
├── database/
│   ├── schema.sql              # Database relational schema
│   ├── db.py                   # SQLite connection & seed account generator
│   └── app.db                  # Local SQLite database file
├── models/
│   └── kmeans_model.pkl        # Pickled K-Means model artifact
├── src/
│   ├── preprocessing/          # cleaner.py (data sanitization & column mapping)
│   ├── analysis/               # eda.py (summary stats, revenue trends, top products)
│   ├── features/               # rfm.py (Recency, Frequency, Monetary calculations)
│   ├── clustering/             # kmeans_model.py (K-Means, Elbow, Silhouette, 2D PCA)
│   └── recommendations/        # engine.py (behavioral personas & marketing playbooks)
├── templates/                  # Bootstrap 5 Jinja2 HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── admin_dashboard.html
│   ├── admin_upload.html
│   ├── admin_preview.html
│   ├── admin_eda.html
│   ├── admin_rfm.html
│   ├── admin_ml.html
│   ├── admin_recommendations.html
│   ├── seller_dashboard.html
│   ├── seller_customers.html
│   └── reports.html
├── static/
│   └── css/style.css           # Custom SaaS stylesheet
├── tests/
│   └── test_pipeline.py        # Automated test suite (6 tests passing)
└── docs/
    ├── ACADEMIC_PROJECT_REPORT.md  # Formal MCA Mini Project Report
    └── VIVA_PREPARATION_GUIDE.md   # Top 15 Viva Q&A, Presentation Script & PPT Outline
```

---

## ⚡ Quickstart Guide

### 1. Activate Environment & Run Application
From PowerShell or Command Prompt in the project folder:
```powershell
cd C:\customer-segmentation-system
.venv\Scripts\python.exe app.py
```

### 2. Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Default Login Credentials
| Role | Username | Password | Purpose |
| :--- | :--- | :--- | :--- |
| **Admin / Analyst** | `admin` | `admin123` | Full access to ML pipeline, upload, EDA, RFM, & K-Means Studio |
| **Seller / Store Owner**| `seller` | `seller123` | Business overview, customer search, export & retention playbooks |

---

## 🧪 Running Automated Tests
To run the automated test suite:
```powershell
.venv\Scripts\python.exe tests\test_pipeline.py
```
*Expected Output: `Ran 6 tests in 0.175s ... OK`*
