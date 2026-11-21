CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'seller', -- 'admin' or 'seller'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    total_rows INTEGER DEFAULT 0,
    total_customers INTEGER DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'uploaded', -- 'uploaded', 'cleaned', 'modeled'
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS customer_rfm (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    customer_id TEXT NOT NULL,
    recency INTEGER NOT NULL,
    frequency INTEGER NOT NULL,
    monetary REAL NOT NULL,
    r_score INTEGER,
    f_score INTEGER,
    m_score INTEGER,
    cluster_id INTEGER,
    segment_label TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
);

CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    algorithm TEXT NOT NULL DEFAULT 'K-Means',
    k_clusters INTEGER NOT NULL,
    silhouette_score REAL,
    inertia REAL,
    model_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
);

CREATE TABLE IF NOT EXISTS marketing_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_label TEXT UNIQUE NOT NULL,
    strategy_title TEXT NOT NULL,
    objective TEXT NOT NULL,
    recommended_channel TEXT NOT NULL,
    suggested_discount TEXT NOT NULL,
    action_plan TEXT NOT NULL
);
