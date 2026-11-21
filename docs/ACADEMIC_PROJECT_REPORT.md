# ACADEMIC MINI PROJECT REPORT

## Title: AI-Based Customer Segmentation and Intelligent Marketing Recommendation System
**Degree / Program:** Master of Computer Applications (MCA)  
**Domain:** Data Science, Machine Learning, & Full-Stack Web Development  
**Academic Year:** 2026  

---

### 1. ABSTRACT
In modern retail and e-commerce enterprises, treating a heterogeneous customer base with uniform, generic marketing campaigns results in suboptimal conversion rates, squandered promotional expenditure, and customer churn. This project implements an intelligent, data-driven analytical web application that transforms raw transactional sales data into actionable business intelligence. The system computes Recency, Frequency, and Monetary (RFM) behavioral vectors per customer, mitigates distributional skewness through logarithmic transformation, normalizes features via standard scaling, and applies unsupervised Machine Learning (K-Means clustering with K-Means++ initialization). Cluster quality is rigorously evaluated using Within-Cluster Sum of Squares (Inertia/Elbow Method) and Silhouette Analysis. Furthermore, cluster centroids are automatically interpreted into human-readable commercial personas (Champions, Loyalists, Potential Loyalists, At-Risk, and Lost Customers), and mapped to automated marketing playbooks. The platform is architected using Python, Flask, SQLite, and Bootstrap 5, featuring specialized dual dashboards for technical Data Analysts and commercial Retail Sellers.

---

### 2. PROBLEM STATEMENT
Traditional retail management suffers from three critical limitations:
1. **Lack of Customer Differentiation:** Marketing teams blast uniform promotional discounts to all registered users regardless of purchasing habits.
2. **Untimely Churn Detection:** Businesses fail to identify previously loyal customers who have entered an inactive or churn state until they are permanently lost.
3. **Absence of Actionable Prescriptions:** Existing Business Intelligence (BI) dashboards display static past metrics (e.g., total sales) without prescribing forward-looking, segment-specific marketing actions or discount recommendations.

---

### 3. OBJECTIVES
- **Automated Data Sanitization:** Ingest large-scale transaction logs (500,000+ records) and eliminate cancellations, returns, guest checkouts, and duplicate rows.
- **Behavioral Feature Engineering:** Aggregate transactional records into multidimensional customer profiles using the RFM framework.
- **Unsupervised Machine Learning:** Implement K-Means clustering to uncover latent, natural groupings within the customer base without requiring manual labeling.
- **Optimal Hyperparameter Selection:** Determine optimal cluster count ($K$) using mathematical heuristics (Elbow WCSS and Silhouette Coefficients).
- **Rule-Based Marketing Prescription:** Translate mathematical centroids into commercial marketing playbooks (channels, discount margins, retention tactics).
- **Dual-Persona Web Deployment:** Provide an intuitive SaaS interface serving both Data Analysts (full ML pipeline controls) and Sellers (non-technical business KPIs).

---

### 4. SYSTEM ARCHITECTURE & DESIGN

#### A. Multi-Tier Architectural Model
```
┌────────────────────────────────────────────────────────┐
│               PRESENTATION LAYER                       │
│  - Bootstrap 5 SaaS Responsive Dashboard               │
│  - Chart.js Dynamic Visualizations (Line, Bar, Donut)  │
│  - 2D PCA Interactive Cluster Projection               │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON
┌───────────────────────────▼────────────────────────────┐
│               APPLICATION LAYER (Flask)                │
│  - Route Controllers (Admin Studio & Seller Center)    │
│  - Session Authentication & RBAC (Admin / Seller)      │
│  - File Download & CSV Stream Exporters                │
└───────────────────────────┬────────────────────────────┘
                            │ Python API
┌───────────────────────────▼────────────────────────────┐
│               DATA SCIENCE & ML ENGINE                 │
│  - Preprocessing: Cleaner & Alias Normalizer           │
│  - Feature Engineering: RFM Derivation & Log Scaling   │
│  - ML Engine: Lloyd's K-Means++ & Silhouette Evaluator │
│  - Recommendation Engine: Centroid-to-Playbook Mapper  │
└───────────────────────────┬────────────────────────────┘
                            │ File & SQL I/O
┌───────────────────────────▼────────────────────────────┐
│               PERSISTENCE LAYER                        │
│  - SQLite Database (users, datasets, strategies)       │
│  - File System (raw CSV, processed CSV, pickled models)│
└────────────────────────────────────────────────────────┘
```

#### B. Entity-Relationship (ER) Description
- **`users` Table:** Stores user credentials (`username`, `email`, `password_hash`, `role`).
- **`datasets` Table:** Tracks uploaded CSV files, row counts, and upload timestamps.
- **`customer_rfm` Table:** Stores customer-level features ($R$, $F$, $M$, quantile scores 1–5, and assigned cluster IDs).
- **`marketing_strategies` Table:** Stores segment playbooks, objectives, channels, and discount guidelines.

---

### 5. MATHEMATICAL METHODOLOGY & ALGORITHMS

#### Step 1: RFM Metric Calculation
Given transaction logs for customer $i$, with transaction dates $t_{i,j}$ and line amounts $a_{i,j} = q_{i,j} \times p_{i,j}$:
$$\text{Snapshot Date } (T_{\text{ref}}) = \max_{i,j}(t_{i,j}) + 1 \text{ day}$$
$$\text{Recency } (R_i) = T_{\text{ref}} - \max_j(t_{i,j}) \quad (\text{days})$$
$$\text{Frequency } (F_i) = \text{Count of unique orders } (\text{Invoices})$$
$$\text{Monetary } (M_i) = \sum_{j} a_{i,j} \quad (\text{Total gross spend})$$

#### Step 2: Distributional Normalization
Transaction monetary and frequency metrics exhibit extreme right-skewness (power-law distribution). To stabilize variance, log transformation is applied:
$$\tilde{x} = \ln(1 + x)$$
Subsequently, features are standardized to zero mean and unit variance:
$$z = \frac{\tilde{x} - \mu}{\sigma}$$

#### Step 3: Lloyd's K-Means Clustering Algorithm
1. **Initialization (K-Means++):** The first centroid $\mu_1$ is chosen uniformly at random from dataset $X$. Subsequent centroids are chosen with probability:
   $$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
2. **Assignment Step (E-step):** Assign each customer vector $x_i$ to the closest centroid:
   $$C_j^{(t)} = \left\{ x_i : \|x_i - \mu_j^{(t)}\|^2 \le \|x_i - \mu_l^{(t)}\|^2 \quad \forall l, 1 \le l \le K \right\}$$
3. **Update Step (M-step):** Calculate new centroids:
   $$\mu_j^{(t+1)} = \frac{1}{|C_j^{(t)}|} \sum_{x_i \in C_j^{(t)}} x_i$$
4. **Convergence:** Repeat until $\|\mu^{(t+1)} - \mu^{(t)}\| < \epsilon$.

#### Step 4: Cluster Quality Validation
- **Within-Cluster Sum of Squares (Inertia):**
  $$\text{WCSS} = \sum_{j=1}^K \sum_{x \in C_j} \|x - \mu_j\|^2$$
- **Silhouette Coefficient ($s_i$):**
  For customer $i \in C_A$:
  $$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \ne i} d(i, j)$$
  $$b(i) = \min_{B \ne A} \frac{1}{|C_B|} \sum_{j \in C_B} d(i, j)$$
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$

---

### 6. EXPERIMENTAL RESULTS ON RETAIL DATASET
Evaluated on the UCI Supermarket transaction corpus (541,910 raw records):
- **Raw Transaction Volume:** 541,910 rows
- **Filtered Guest Checkouts (Null Customer IDs):** 135,080 rows
- **Filtered Cancellations & Negative Prices:** 8,912 rows
- **Cleaned Working Transactions:** 392,693 rows
- **Unique Customers Segmented:** 4,338 customers
- **Total Gross Revenue Analyzed:** $8,887,226.89

#### Optimal Cluster Segmentation Profile ($K=4$):
| Cluster ID | Assigned Segment Label | Customer Count | Avg Recency (Days) | Avg Orders | Avg Spend ($) | Total Revenue Share ($) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Champions / VIPs** | 646 | 14.8 | 15.2 | $8,433.11 | $5,447,790.58 (61.3%) |
| **3** | **Loyal Customers** | 1,169 | 42.1 | 5.8 | $2,034.80 | $2,378,686.44 (26.8%) |
| **2** | **Potential Loyalists** | 872 | 58.4 | 2.1 | $535.08 | $466,592.93 (5.2%) |
| **1** | **Lost / Inactive** | 1,651 | 247.9 | 1.3 | $359.88 | $594,156.94 (6.7%) |

**Key Business Insight:** Just 14.8% of the customer base (Champions) drive over 61% of total enterprise sales, demonstrating the classic Pareto Principle (80/20 rule) in commercial retailing.

---

### 7. LIMITATIONS & FUTURE SCOPE
- **Static Batch Processing:** Current pipeline processes uploaded CSV files periodically; future enhancements can integrate Kafka or RabbitMQ for real-time streaming transaction updates.
- **Deep Learning Embeddings:** Beyond RFM, future iterations can incorporate natural language processing (e.g., Sentence-BERT) on product descriptions to model semantic item affinities.
- **Automated Communication Triggers:** Directly integrate SMTP and WhatsApp Cloud API to trigger automated promotional emails when a customer crosses segment thresholds.

---

### 8. CONCLUSION
The AI-Based Customer Segmentation and Intelligent Marketing Recommendation System demonstrates the successful convergence of Data Engineering, Unsupervised Machine Learning, and Full-Stack Web Development. By translating raw e-commerce transaction logs into mathematical RFM vectors and dynamic commercial marketing playbooks, the platform empowers enterprise stakeholders to maximize marketing ROI, nurture high-value loyalty, and proactively combat customer churn.
