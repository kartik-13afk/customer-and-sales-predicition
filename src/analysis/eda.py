import os
import pandas as pd
import numpy as np

def generate_eda_summary(df):
    """
    Computes key exploratory data analysis metrics from cleaned transaction data.
    """
    total_revenue = round(float(df['total_amount'].sum()), 2)
    total_transactions = int(df['invoice_no'].nunique())
    unique_customers = int(df['customer_id'].nunique())
    avg_order_value = round(total_revenue / total_transactions, 2) if total_transactions > 0 else 0.0
    avg_customer_spend = round(total_revenue / unique_customers, 2) if unique_customers > 0 else 0.0

    # 1. Top 10 products by quantity sold
    top_products_qty = df.groupby('description')['quantity'].sum().sort_values(ascending=False).head(10)
    top_products_list = [{'product': str(k), 'quantity': int(v)} for k, v in top_products_qty.items()]

    # 2. Top 10 products by revenue generated
    top_products_rev = df.groupby('description')['total_amount'].sum().sort_values(ascending=False).head(10)
    top_revenue_products_list = [{'product': str(k), 'revenue': round(float(v), 2)} for k, v in top_products_rev.items()]

    # 3. Country-wise sales distribution (Top 10 countries)
    country_sales = df.groupby('country').agg(
        total_revenue=('total_amount', 'sum'),
        transactions=('invoice_no', 'nunique'),
        customers=('customer_id', 'nunique')
    ).sort_values(by='total_revenue', ascending=False).head(10)

    country_sales_list = []
    for country, row in country_sales.iterrows():
        country_sales_list.append({
            'country': str(country),
            'revenue': round(float(row['total_revenue']), 2),
            'transactions': int(row['transactions']),
            'customers': int(row['customers'])
        })

    # 4. Monthly revenue trends
    if 'invoice_date' in df.columns:
        df_temp = df.copy()
        df_temp['year_month'] = pd.to_datetime(df_temp['invoice_date']).dt.to_period('M').astype(str)
        monthly_trend = df_temp.groupby('year_month')['total_amount'].sum().sort_index()
        monthly_labels = list(monthly_trend.index)
        monthly_values = [round(float(v), 2) for v in monthly_trend.values]
    else:
        monthly_labels, monthly_values = [], []

    return {
        'kpis': {
            'total_revenue': total_revenue,
            'total_transactions': total_transactions,
            'unique_customers': unique_customers,
            'avg_order_value': avg_order_value,
            'avg_customer_spend': avg_customer_spend
        },
        'top_products_by_quantity': top_products_list,
        'top_products_by_revenue': top_revenue_products_list,
        'country_breakdown': country_sales_list,
        'monthly_revenue_trend': {
            'labels': monthly_labels,
            'data': monthly_values
        }
    }

if __name__ == '__main__':
    cleaned_path = r'C:\customer-segmentation-system\data\processed\cleaned_transactions.csv'
    if os.path.exists(cleaned_path):
        print("[TEST] Running EDA on cleaned_transactions.csv...")
        df = pd.read_csv(cleaned_path)
        summary = generate_eda_summary(df)
        print("[REPORT] EDA Key Performance Indicators:")
        for k, v in summary['kpis'].items():
            print(f"  - {k}: {v:,}" if isinstance(v, (int, float)) else f"  - {k}: {v}")
        print("\nTop 3 Countries by Revenue:")
        for c in summary['country_breakdown'][:3]:
            print(f"  - {c['country']}:  across {c['transactions']:,} orders")
