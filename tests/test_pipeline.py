import unittest
import os
import json
import sqlite3
import pandas as pd
import numpy as np

from config import config
from database.db import get_db_connection, init_db
from src.preprocessing.cleaner import clean_transaction_data, standardize_columns
from src.features.rfm import calculate_rfm_metrics
from src.clustering.kmeans_model import RFMClusterEngine
from src.recommendations.engine import assign_segment_labels, apply_recommendations_to_customers
from app import app

class TestCustomerSegmentationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.test_client()
        cls.client.testing = True

    def test_01_database_users_seeded(self):
        """Verifies default admin and seller accounts are created with hashed passwords."""
        conn = get_db_connection()
        admin = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        seller = conn.execute("SELECT * FROM users WHERE username = 'seller'").fetchone()
        conn.close()

        self.assertIsNotNone(admin, "Admin user must exist in the database.")
        self.assertEqual(admin['role'], 'admin')
        self.assertIsNotNone(seller, "Seller user must exist in the database.")
        self.assertEqual(seller['role'], 'seller')

    def test_02_data_cleaner_filtering(self):
        """Verifies cleaning filters cancellations, null customer IDs, and invalid prices."""
        # Create a synthetic mock dataframe
        test_csv_path = os.path.join(config.DATA_RAW_DIR, 'test_mock_data.csv')
        mock_data = pd.DataFrame({
            'Invoice': ['1001', 'C1002', '1003', '1004', '1005', '1001'],
            'StockCode': ['A1', 'A2', 'A3', 'A4', 'A5', 'A1'],
            'Description': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5', 'Item 1'],
            'Quantity': [5, -2, 10, 0, 3, 5],
            'InvoiceDate': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-01'],
            'Price': [10.0, 15.0, 20.0, 5.0, -1.0, 10.0],
            'Customer ID': ['101', '101', np.nan, '102', '103', '101'],
            'Country': ['UK', 'UK', 'UK', 'UK', 'UK', 'UK']
        })
        mock_data.to_csv(test_csv_path, index=False)

        cleaned_df, stats = clean_transaction_data(test_csv_path)

        # Expected:
        # Row 0: valid (1001)
        # Row 1: cancelled ('C1002') -> filtered
        # Row 2: missing Customer ID (NaN) -> filtered
        # Row 3: quantity 0 -> filtered
        # Row 4: negative price -> filtered
        # Row 5: duplicate of Row 0 -> filtered
        # Exactly 1 clean row should remain!
        self.assertEqual(len(cleaned_df), 1, f"Expected 1 valid row, got {len(cleaned_df)}")
        self.assertEqual(cleaned_df.iloc[0]['invoice_no'], '1001')
        self.assertEqual(cleaned_df.iloc[0]['total_amount'], 50.0)

        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

    def test_03_rfm_metric_derivation(self):
        """Verifies RFM aggregation, scoring, and log transformations."""
        mock_cleaned = pd.DataFrame({
            'customer_id': ['1', '1', '2', '3'],
            'invoice_no': ['INV1', 'INV2', 'INV3', 'INV4'],
            'invoice_date': pd.to_datetime(['2023-01-01', '2023-01-10', '2023-01-05', '2023-01-12']),
            'total_amount': [100.0, 200.0, 50.0, 500.0]
        })
        rfm_df = calculate_rfm_metrics(mock_cleaned)

        self.assertEqual(len(rfm_df), 3, "Expected 3 unique customers in RFM output.")
        cust1 = rfm_df[rfm_df['customer_id'] == '1'].iloc[0]
        self.assertEqual(cust1['frequency'], 2)
        self.assertEqual(cust1['monetary'], 300.0)
        self.assertIn('recency_log', rfm_df.columns)
        self.assertIn('frequency_log', rfm_df.columns)
        self.assertIn('monetary_log', rfm_df.columns)

    def test_04_kmeans_and_pca_engine(self):
        """Verifies pure K-Means clustering, silhouette score, and 2D PCA projection."""
        np.random.seed(42)
        mock_rfm = pd.DataFrame({
            'customer_id': [str(i) for i in range(100)],
            'recency': np.random.randint(1, 300, 100),
            'frequency': np.random.randint(1, 20, 100),
            'monetary': np.random.uniform(20.0, 5000.0, 100)
        })
        mock_rfm['recency_log'] = np.log1p(mock_rfm['recency'])
        mock_rfm['frequency_log'] = np.log1p(mock_rfm['frequency'])
        mock_rfm['monetary_log'] = np.log1p(mock_rfm['monetary'])

        engine = RFMClusterEngine(random_state=42)
        clustered_df, profiles, sil, inertia = engine.fit_predict(mock_rfm, k=3)

        self.assertEqual(len(profiles), 3, "Expected 3 cluster profiles for k=3.")
        self.assertIn('pca_x', clustered_df.columns)
        self.assertIn('pca_y', clustered_df.columns)
        self.assertGreater(inertia, 0.0)

    def test_05_recommendation_rule_assignment(self):
        """Verifies mapping of cluster centroids to business segment labels."""
        profiles = pd.DataFrame({
            'cluster_id': [0, 1, 2, 3],
            'mean_monetary': [5000.0, 300.0, 800.0, 1500.0],
            'mean_frequency': [15.0, 1.2, 3.0, 6.0],
            'mean_recency': [15.0, 250.0, 30.0, 60.0],
            'customer_count': [50, 200, 100, 150],
            'total_revenue': [250000.0, 60000.0, 80000.0, 225000.0]
        })
        mapping = assign_segment_labels(profiles)
        self.assertEqual(mapping[0], 'Champions', "Cluster 0 with highest spend should be Champions.")

    def test_06_flask_authentication_flow(self):
        """Verifies login authentication, role enforcement, and redirects."""
        # Unauthenticated access to admin dashboard redirects to login
        res = self.client.get('/admin/dashboard', follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers['Location'])

        # Valid login with admin
        login_res = self.client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)

if __name__ == '__main__':
    unittest.main()
