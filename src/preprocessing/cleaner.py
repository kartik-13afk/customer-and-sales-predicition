import os
import pandas as pd
import numpy as np

COLUMN_MAPPINGS = {
    # Invoice / Order identifiers
    'invoiceno': 'invoice_no',
    'invoice': 'invoice_no',
    'invoicenumber': 'invoice_no',
    'orderid': 'invoice_no',
    'order_id': 'invoice_no',
    'orderno': 'invoice_no',
    'transactionid': 'invoice_no',
    'transaction_id': 'invoice_no',
    
    # Stock / Product identifiers
    'stockcode': 'stock_code',
    'stock_code': 'stock_code',
    'productid': 'stock_code',
    'product_id': 'stock_code',
    'itemcode': 'stock_code',
    'item_code': 'stock_code',
    'sku': 'stock_code',
    
    # Description / Product Name
    'description': 'description',
    'productname': 'description',
    'product_name': 'description',
    'product': 'description',
    'item': 'description',
    'item_name': 'description',
    
    # Quantity
    'quantity': 'quantity',
    'qty': 'quantity',
    'units': 'quantity',
    'count': 'quantity',
    
    # Date
    'invoicedate': 'invoice_date',
    'invoice_date': 'invoice_date',
    'orderdate': 'invoice_date',
    'order_date': 'invoice_date',
    'date': 'invoice_date',
    'timestamp': 'invoice_date',
    
    # Price
    'unitprice': 'unit_price',
    'unit_price': 'unit_price',
    'price': 'unit_price',
    'rate': 'unit_price',
    'itemprice': 'unit_price',
    'item_price': 'unit_price',
    'salesprice': 'unit_price',
    
    # Customer
    'customerid': 'customer_id',
    'customer_id': 'customer_id',
    'customer': 'customer_id',
    'clientid': 'customer_id',
    'userid': 'customer_id',
    'user_id': 'customer_id',
    'custid': 'customer_id',
    
    # Country / Region
    'country': 'country',
    'region': 'country',
    'location': 'country',
    'city': 'country',
    
    # Direct total
    'total': 'total_amount',
    'totalamount': 'total_amount',
    'total_amount': 'total_amount',
    'sales': 'total_amount',
    'revenue': 'total_amount'
}

def standardize_columns(df):
    clean_cols = {}
    for col in df.columns:
        normalized = str(col).strip().lower().replace('_', '').replace(' ', '')
        clean_cols[col] = COLUMN_MAPPINGS.get(normalized, str(col).strip().lower())
    df = df.rename(columns=clean_cols)
    return df

def clean_transaction_data(file_path):
    """
    Cleans raw retail transaction CSV with universal column alias support.
    """
    # 1. Load data
    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    raw_row_count = len(df)
    
    # 2. Standardize column names
    df = standardize_columns(df)
    
    # Audit tracking
    stats = {
        'raw_rows': raw_row_count,
        'missing_customers': 0,
        'cancelled_orders': 0,
        'invalid_quantity': 0,
        'invalid_price': 0,
        'duplicate_rows': 0,
        'final_rows': 0,
        'unique_customers': 0,
        'total_revenue': 0.0
    }
    
    # 3. Handle Missing Customer IDs
    if 'customer_id' in df.columns:
        missing_cust_mask = df['customer_id'].isnull()
        stats['missing_customers'] = int(missing_cust_mask.sum())
        df = df[~missing_cust_mask].copy()
        df['customer_id'] = df['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    else:
        # If no customer ID column is found, synthesize an index-based customer for resilience
        df['customer_id'] = [f"CUST_{i%500 + 1}" for i in range(len(df))]
        
    # 4. Handle duplicates
    dup_count = int(df.duplicated().sum())
    stats['duplicate_rows'] = dup_count
    df = df.drop_duplicates()
    
    # 5. Handle Cancelled Orders (Invoices starting with 'C')
    if 'invoice_no' in df.columns:
        df['invoice_no'] = df['invoice_no'].astype(str)
        cancellation_mask = df['invoice_no'].str.startswith('C', na=False)
        stats['cancelled_orders'] = int(cancellation_mask.sum())
        df = df[~cancellation_mask]
    else:
        df['invoice_no'] = [f"INV_{i+1}" for i in range(len(df))]
        
    # 6. Filter positive Quantity and Unit Price
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        invalid_qty = (df['quantity'] <= 0) | df['quantity'].isnull()
        stats['invalid_quantity'] = int(invalid_qty.sum())
        df = df[~invalid_qty]
    else:
        df['quantity'] = 1
        
    if 'unit_price' in df.columns:
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        invalid_price = (df['unit_price'] <= 0) | df['unit_price'].isnull()
        stats['invalid_price'] = int(invalid_price.sum())
        df = df[~invalid_price]
    else:
        if 'total_amount' in df.columns:
            df['unit_price'] = df['total_amount']
        else:
            df['unit_price'] = 10.0
            
    # 7. Date parsing
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['invoice_date'])
    else:
        df['invoice_date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='h')
        
    # 8. Compute Total Amount (Revenue per transaction line)
    if 'total_amount' not in df.columns or df['total_amount'].isnull().all():
        df['total_amount'] = (df['quantity'] * df['unit_price']).round(2)
    else:
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(df['quantity'] * df['unit_price']).round(2)
        
    if 'country' not in df.columns:
        df['country'] = 'General'
        
    if 'description' not in df.columns:
        df['description'] = 'Retail Product'
        
    if 'stock_code' not in df.columns:
        df['stock_code'] = 'ITEM'
        
    # Final statistics
    stats['final_rows'] = len(df)
    stats['unique_customers'] = int(df['customer_id'].nunique())
    stats['total_revenue'] = round(float(df['total_amount'].sum()), 2)
    
    return df, stats
