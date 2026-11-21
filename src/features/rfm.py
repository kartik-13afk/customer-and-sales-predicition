import os
import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_rfm_metrics(df):
    """
    Computes Recency, Frequency, and Monetary (RFM) metrics per customer.
    Args:
        df (pd.DataFrame): Cleaned transaction DataFrame with 'customer_id', 'invoice_no', 'invoice_date', 'total_amount'.
    Returns:
        rfm_df (pd.DataFrame): Customer-level RFM features.
    """
    df = df.copy()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    # Snapshot date: 1 day after the latest recorded transaction
    snapshot_date = df['invoice_date'].max() + timedelta(days=1)
    
    # Customer-level aggregation
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (snapshot_date - x.max()).days,
        'invoice_no': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    
    # Rename columns to standard RFM definitions
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Ensure numerical safety (monetary must be >= 0)
    rfm['monetary'] = rfm['monetary'].round(2)
    rfm = rfm[rfm['monetary'] > 0].copy()
    
    # RFM Scoring using quantiles (1 to 5)
    # Recency: Lower is better -> inverted labels
    r_labels = [5, 4, 3, 2, 1]
    f_labels = [1, 2, 3, 4, 5]
    m_labels = [1, 2, 3, 4, 5]
    
    try:
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=r_labels).astype(int)
    except ValueError:
        # Fallback to rank method if bins have duplicate edges
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), q=5, labels=r_labels).astype(int)
        
    try:
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=f_labels).astype(int)
    except ValueError:
        rfm['f_score'] = pd.cut(rfm['frequency'], bins=5, labels=f_labels).astype(int)
        
    try:
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=m_labels).astype(int)
    except ValueError:
        rfm['m_score'] = pd.cut(rfm['monetary'], bins=5, labels=m_labels).astype(int)
        
    rfm['rfm_combined'] = (
        rfm['r_score'].astype(str) + 
        rfm['f_score'].astype(str) + 
        rfm['m_score'].astype(str)
    )
    
    # Transformations for Machine Learning
    # Log transformation reduces skewness
    rfm['recency_log'] = np.log1p(rfm['recency'])
    rfm['frequency_log'] = np.log1p(rfm['frequency'])
    rfm['monetary_log'] = np.log1p(rfm['monetary'])
    
    return rfm

if __name__ == '__main__':
    cleaned_path = r'C:\customer-segmentation-system\data\processed\cleaned_transactions.csv'
    if os.path.exists(cleaned_path):
        print("[TEST] Computing RFM on cleaned_transactions.csv...")
        df = pd.read_csv(cleaned_path)
        rfm_df = calculate_rfm_metrics(df)
        print(f"[SUCCESS] Calculated RFM for {len(rfm_df):,} customers.")
        print("\nFirst 5 customer RFM records:")
        print(rfm_df[['customer_id', 'recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score']].head())
        out_path = r'C:\customer-segmentation-system\data\processed\customer_rfm.csv'
        rfm_df.to_csv(out_path, index=False)
        print(f"\n[SAVED] Saved customer RFM table to: {out_path}")
