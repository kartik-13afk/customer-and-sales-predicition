import os
import io
import csv
import json
import pandas as pd
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, session, send_file, Response, jsonify
)
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from config import config
from database.db import get_db_connection, init_db
from src.preprocessing.cleaner import clean_transaction_data
from src.analysis.eda import generate_eda_summary
from src.features.rfm import calculate_rfm_metrics
from src.clustering.kmeans_model import RFMClusterEngine
from src.recommendations.engine import (
    assign_segment_labels, apply_recommendations_to_customers, 
    get_segment_strategy_summary, MARKETING_PLAYBOOKS
)

app = Flask(__name__)
app.config.from_object(config)

# WSGI Middleware to handle Vercel serverless routing
class VercelPathFixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path.startswith('/api/index') or path == '/api':
            # Vercel passes the original URL in HTTP_X_MATCHED_PATH or HTTP_X_FORWARDED_URI
            matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
            if matched:
                environ['PATH_INFO'] = matched
            else:
                environ['PATH_INFO'] = '/'
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

# If running on Vercel serverless, sync precomputed assets to writable /tmp
if getattr(config, 'IS_VERCEL', False):
    import shutil
    for folder in ['data/processed', 'models', 'database']:
        src = os.path.join(config.BASE_DIR, folder)
        dest = os.path.join('/tmp', folder)
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.copytree(src, dest, dirs_exist_ok=True)

# Ensure required directories exist
for path in [config.DATA_RAW_DIR, config.DATA_PROCESSED_DIR, config.MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

# Initialize database tables & seed data on startup
with app.app_context():
    init_db()

# --- Authentication Helpers ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please sign in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Administrative privileges required to access this resource.', 'danger')
            return redirect(url_for('seller_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Global Pipeline Helpers ---
def get_processed_files():
    cleaned_path = os.path.join(config.DATA_PROCESSED_DIR, 'cleaned_transactions.csv')
    rfm_path = os.path.join(config.DATA_PROCESSED_DIR, 'customer_rfm.csv')
    segmented_path = os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')
    model_path = os.path.join(config.MODELS_DIR, 'kmeans_model.pkl')
    return cleaned_path, rfm_path, segmented_path, model_path

def run_full_pipeline(raw_file_path, k=4):
    """
    Executes the complete Data Science & ML pipeline end-to-end.
    """
    cleaned_path, rfm_path, segmented_path, model_path = get_processed_files()
    
    # 1. Clean data
    cleaned_df, cleaning_stats = clean_transaction_data(raw_file_path)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # Save audit stats to JSON for quick retrieval
    stats_json_path = os.path.join(config.DATA_PROCESSED_DIR, 'cleaning_stats.json')
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaning_stats, f)
        
    # 2. Compute RFM
    rfm_df = calculate_rfm_metrics(cleaned_df)
    rfm_df.to_csv(rfm_path, index=False)
    
    # 3. Clustering
    engine = RFMClusterEngine(random_state=42)
    eval_results = engine.evaluate_k(rfm_df)
    eval_json_path = os.path.join(config.DATA_PROCESSED_DIR, 'ml_eval.json')
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f)
        
    clustered_df, profiles, sil, inertia = engine.fit_predict(rfm_df, k=k, model_save_path=model_path)
    
    # 4. Recommendation Mapping
    cluster_mapping = assign_segment_labels(profiles)
    final_df = apply_recommendations_to_customers(clustered_df, cluster_mapping)
    final_df.to_csv(segmented_path, index=False)
    
    # Save segment strategy summary
    strategies = get_segment_strategy_summary(cluster_mapping, profiles)
    strat_json_path = os.path.join(config.DATA_PROCESSED_DIR, 'segment_strategies.json')
    with open(strat_json_path, 'w', encoding='utf-8') as f:
        json.dump(strategies, f)
        
    # Also save model summary
    model_summary = {
        'k': k,
        'silhouette_score': sil,
        'inertia': inertia,
        'profiles': profiles.to_dict(orient='records')
    }
    with open(os.path.join(config.DATA_PROCESSED_DIR, 'model_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(model_summary, f)
        
    return cleaning_stats, final_df, strategies

# --- Auth Routes ---
@app.route('/')
def index():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('seller_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'Welcome back, {user["username"]}!', 'success')
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('seller_dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        role = request.form.get('role', 'seller').strip()
        password = request.form.get('password', '').strip()

        if not username or not password or not email:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        conn = get_db_connection()
        existing = conn.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email)).fetchone()
        if existing:
            conn.close()
            flash('Username or email already taken.', 'warning')
            return redirect(url_for('register'))

        p_hash = generate_password_hash(password)
        conn.execute("INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                     (username, email, p_hash, role))
        conn.commit()
        conn.close()

        flash('Registration successful! You may now sign in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out safely.', 'info')
    return redirect(url_for('login'))

# --- Admin Routes ---
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats_json = os.path.join(config.DATA_PROCESSED_DIR, 'cleaning_stats.json')
    strat_json = os.path.join(config.DATA_PROCESSED_DIR, 'segment_strategies.json')
    model_json = os.path.join(config.DATA_PROCESSED_DIR, 'model_summary.json')

    stats = None
    segment_summary = []
    k_clusters = 4

    if os.path.exists(stats_json):
        with open(stats_json, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    if os.path.exists(strat_json):
        with open(strat_json, 'r', encoding='utf-8') as f:
            segment_summary = json.load(f)
    if os.path.exists(model_json):
        with open(model_json, 'r', encoding='utf-8') as f:
            m_info = json.load(f)
            k_clusters = m_info.get('k', 4)

    return render_template(
        'admin_dashboard.html',
        active_page='admin_dashboard',
        stats=stats,
        segment_summary=segment_summary,
        k_clusters=k_clusters
    )

@app.route('/admin/upload', methods=['GET', 'POST'])
@admin_required
def admin_upload():
    if request.method == 'POST':
        if 'dataset_file' not in request.files:
            flash('No file part selected.', 'warning')
            return redirect(request.url)
        file = request.files['dataset_file']
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            fname = secure_filename(file.filename)
            save_path = os.path.join(config.DATA_RAW_DIR, fname)
            file.save(save_path)
            flash(f'File "{fname}" uploaded successfully. Running data science pipeline...', 'info')
            try:
                run_full_pipeline(save_path)
                flash('Pipeline execution completed! Preprocessing, RFM, and ML models are ready.', 'success')
                return redirect(url_for('admin_preview'))
            except Exception as e:
                flash(f'Error processing dataset: {str(e)}', 'danger')
        else:
            flash('Only CSV format files are supported.', 'danger')

    return render_template('admin_upload.html', active_page='admin_upload')

@app.route('/admin/load-sample', methods=['POST'])
@admin_required
def load_sample_dataset():
    sample_path = os.path.join(config.DATA_RAW_DIR, 'supermarket_data.csv')
    if not os.path.exists(sample_path):
        flash('Default sample supermarket dataset not found.', 'danger')
        return redirect(url_for('admin_upload'))
    try:
        flash('Loading sample retail dataset and executing end-to-end pipeline...', 'info')
        run_full_pipeline(sample_path)
        flash('Sample dataset loaded & processed successfully!', 'success')
        return redirect(url_for('admin_preview'))
    except Exception as e:
        flash(f'Error running pipeline on sample dataset: {str(e)}', 'danger')
        return redirect(url_for('admin_upload'))

@app.route('/admin/preview')
@admin_required
def admin_preview():
    cleaned_path = os.path.join(config.DATA_PROCESSED_DIR, 'cleaned_transactions.csv')
    stats_json = os.path.join(config.DATA_PROCESSED_DIR, 'cleaning_stats.json')

    if not os.path.exists(cleaned_path):
        flash('Please upload or load a dataset first.', 'warning')
        return redirect(url_for('admin_upload'))

    stats = None
    if os.path.exists(stats_json):
        with open(stats_json, 'r', encoding='utf-8') as f:
            stats = json.load(f)

    # Read first 15 rows for inspection
    df_sample = pd.read_csv(cleaned_path, nrows=15)
    preview_rows = df_sample.to_dict(orient='records')

    return render_template(
        'admin_preview.html',
        active_page='admin_preview',
        stats=stats,
        preview_rows=preview_rows
    )

@app.route('/admin/eda')
@admin_required
def admin_eda():
    cleaned_path = os.path.join(config.DATA_PROCESSED_DIR, 'cleaned_transactions.csv')
    if not os.path.exists(cleaned_path):
        flash('Please process a dataset first.', 'warning')
        return redirect(url_for('admin_upload'))

    eda_cache = os.path.join(config.DATA_PROCESSED_DIR, 'eda_summary.json')
    if os.path.exists(eda_cache):
        with open(eda_cache, 'r', encoding='utf-8') as f:
            eda = json.load(f)
    else:
        df = pd.read_csv(cleaned_path)
        eda = generate_eda_summary(df)
        with open(eda_cache, 'w', encoding='utf-8') as f:
            json.dump(eda, f)

    return render_template('admin_eda.html', active_page='admin_eda', eda=eda)

@app.route('/admin/rfm')
@admin_required
def admin_rfm():
    rfm_path = os.path.join(config.DATA_PROCESSED_DIR, 'customer_rfm.csv')
    if not os.path.exists(rfm_path):
        flash('RFM metrics not yet generated. Please process a dataset.', 'warning')
        return redirect(url_for('admin_upload'))

    rfm_df = pd.read_csv(rfm_path, nrows=30)
    rfm_rows = rfm_df.to_dict(orient='records')

    return render_template('admin_rfm.html', active_page='admin_rfm', rfm_rows=rfm_rows)

@app.route('/admin/ml')
@admin_required
def admin_ml():
    model_json = os.path.join(config.DATA_PROCESSED_DIR, 'model_summary.json')
    eval_json = os.path.join(config.DATA_PROCESSED_DIR, 'ml_eval.json')
    segmented_path = os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')

    if not os.path.exists(model_json) or not os.path.exists(segmented_path):
        flash('ML models not trained yet. Please process a dataset first.', 'warning')
        return redirect(url_for('admin_upload'))

    with open(model_json, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    with open(eval_json, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    # Sample PCA points for visual scatter plot (max 400 points for crisp frontend rendering)
    seg_df = pd.read_csv(segmented_path)
    sample_df = seg_df.sample(n=min(len(seg_df), 400), random_state=42)
    pca_sample = sample_df[['cluster_id', 'pca_x', 'pca_y']].to_dict(orient='records')

    return render_template(
        'admin_ml.html',
        active_page='admin_ml',
        active_k=model_info.get('k', 4),
        sil_score=model_info.get('silhouette_score', 0.0),
        inertia=model_info.get('inertia', 0.0),
        profiles=model_info.get('profiles', []),
        eval_results=eval_results,
        pca_sample=pca_sample
    )

@app.route('/admin/ml/retrain', methods=['POST'])
@admin_required
def retrain_model():
    k = int(request.form.get('k_clusters', 4))
    rfm_path = os.path.join(config.DATA_PROCESSED_DIR, 'customer_rfm.csv')
    model_path = os.path.join(config.MODELS_DIR, 'kmeans_model.pkl')
    segmented_path = os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')

    if not os.path.exists(rfm_path):
        flash('RFM dataset missing. Please upload a dataset first.', 'warning')
        return redirect(url_for('admin_upload'))

    rfm_df = pd.read_csv(rfm_path)
    engine = RFMClusterEngine(random_state=42)
    clustered_df, profiles, sil, inertia = engine.fit_predict(rfm_df, k=k, model_save_path=model_path)

    cluster_mapping = assign_segment_labels(profiles)
    final_df = apply_recommendations_to_customers(clustered_df, cluster_mapping)
    final_df.to_csv(segmented_path, index=False)

    strategies = get_segment_strategy_summary(cluster_mapping, profiles)
    with open(os.path.join(config.DATA_PROCESSED_DIR, 'segment_strategies.json'), 'w', encoding='utf-8') as f:
        json.dump(strategies, f)

    model_summary = {
        'k': k,
        'silhouette_score': sil,
        'inertia': inertia,
        'profiles': profiles.to_dict(orient='records')
    }
    with open(os.path.join(config.DATA_PROCESSED_DIR, 'model_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(model_summary, f)

    flash(f'Model retrained successfully with K={k}! Silhouette Score: {sil}', 'success')
    return redirect(url_for('admin_ml'))

@app.route('/admin/recommendations')
@admin_required
def admin_recommendations():
    strat_json = os.path.join(config.DATA_PROCESSED_DIR, 'segment_strategies.json')
    if not os.path.exists(strat_json):
        flash('Recommendations not yet computed.', 'warning')
        return redirect(url_for('admin_upload'))

    with open(strat_json, 'r', encoding='utf-8') as f:
        strategies = json.load(f)

    return render_template('admin_recommendations.html', active_page='admin_recommendations', strategies=strategies)

# --- Seller Routes ---
@app.route('/seller/dashboard')
@login_required
def seller_dashboard():
    strat_json = os.path.join(config.DATA_PROCESSED_DIR, 'segment_strategies.json')
    stats_json = os.path.join(config.DATA_PROCESSED_DIR, 'cleaning_stats.json')

    if not os.path.exists(strat_json):
        flash('No analytics data available yet. Please ask an Admin to upload a dataset.', 'info')
        return render_template(
            'seller_dashboard.html',
            active_page='seller_dashboard',
            total_revenue=0.0,
            total_customers=0,
            avg_order_value=0.0,
            champions_count=0,
            opportunities=[],
            segment_summary=[]
        )

    with open(strat_json, 'r', encoding='utf-8') as f:
        segment_summary = json.load(f)

    stats = {}
    if os.path.exists(stats_json):
        with open(stats_json, 'r', encoding='utf-8') as f:
            stats = json.load(f)

    total_revenue = stats.get('total_revenue', 0.0)
    total_customers = stats.get('unique_customers', 0)
    avg_order_value = round(total_revenue / stats.get('final_rows', 1), 2) if stats.get('final_rows', 0) > 0 else 0.0

    champions_count = 0
    opportunities = []

    for s in segment_summary:
        if s['segment_label'] == 'Champions':
            champions_count = s['customer_count']
        opportunities.append({
            'segment_label': s['segment_label'],
            'badge_color': s['badge_color'],
            'title': s['strategy_title'],
            'summary': s['objective'],
            'offer': s['suggested_discount'],
            'count': s['customer_count']
        })

    return render_template(
        'seller_dashboard.html',
        active_page='seller_dashboard',
        total_revenue=total_revenue,
        total_customers=total_customers,
        avg_order_value=avg_order_value,
        champions_count=champions_count,
        opportunities=opportunities,
        segment_summary=segment_summary
    )

@app.route('/seller/customers')
@login_required
def seller_customers():
    segmented_path = os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')
    if not os.path.exists(segmented_path):
        flash('Customer data not yet available.', 'warning')
        return render_template('seller_customers.html', active_page='seller_customers', customers=[], segment_options=[])

    df = pd.read_csv(segmented_path)
    segment_options = sorted(list(df['segment_label'].dropna().unique()))

    active_segment = request.args.get('segment', '').strip()
    search_query = request.args.get('search', '').strip()

    filtered_df = df.copy()
    if active_segment:
        filtered_df = filtered_df[filtered_df['segment_label'] == active_segment]
    if search_query:
        filtered_df = filtered_df[filtered_df['customer_id'].astype(str).str.contains(search_query, na=False)]

    total_count = len(df)
    # Return top 100 matching rows
    display_rows = filtered_df.head(100).to_dict(orient='records')

    return render_template(
        'seller_customers.html',
        active_page='seller_customers',
        customers=display_rows,
        segment_options=segment_options,
        active_segment=active_segment,
        search_query=search_query,
        total_count=total_count
    )

# --- Reports & File Downloads ---
@app.route('/reports')
@login_required
def reports_view():
    return render_template('reports.html', active_page='reports')

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    allowed_files = {
        'cleaned_transactions.csv': os.path.join(config.DATA_PROCESSED_DIR, 'cleaned_transactions.csv'),
        'customer_rfm.csv': os.path.join(config.DATA_PROCESSED_DIR, 'customer_rfm.csv'),
        'final_customer_segments.csv': os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')
    }
    if filename not in allowed_files or not os.path.exists(allowed_files[filename]):
        flash('Requested file is not available.', 'danger')
        return redirect(url_for('reports_view'))
    return send_file(allowed_files[filename], as_attachment=True, download_name=filename)

@app.route('/export/customers')
@login_required
def export_customers_csv():
    segmented_path = os.path.join(config.DATA_PROCESSED_DIR, 'final_customer_segments.csv')
    if not os.path.exists(segmented_path):
        flash('No customer data available for export.', 'warning')
        return redirect(url_for('seller_customers'))

    df = pd.read_csv(segmented_path)
    seg = request.args.get('segment', '').strip()
    if seg:
        df = df[df['segment_label'] == seg]

    output = io.StringIO()
    cols_to_export = ['customer_id', 'segment_label', 'recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score', 'marketing_title']
    cols_present = [c for c in cols_to_export if c in df.columns]
    df[cols_present].to_csv(output, index=False)

    fname = f"customers_{seg.lower().replace(' ', '_') if seg else 'all'}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={fname}"}
    )

if __name__ == '__main__':
    # Initial sample check: if processed files don't exist yet, process raw sample dataset
    cleaned_file, _, seg_file, _ = get_processed_files()
    raw_sample = os.path.join(config.DATA_RAW_DIR, 'supermarket_data.csv')
    if not os.path.exists(seg_file) and os.path.exists(raw_sample):
        print("[INIT] First-time startup: Running initial data science pipeline on sample data...")
        run_full_pipeline(raw_sample)
        print("[INIT] Startup pipeline complete!")

    print("\n" + "="*60)
    print("🚀 AI Customer Segmentation Platform is Ready!")
    print("📍 Local Web URL: http://127.0.0.1:5000")
    print("👤 Admin Login:  admin  / admin123")
    print("👤 Seller Login: seller / seller123")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
