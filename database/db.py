import sqlite3
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from werkzeug.security import generate_password_hash
from config import config

def get_db_connection():
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r', encoding='utf-8') as f:
        conn.executescript(f.read())
    conn.commit()
    seed_initial_data(conn)
    conn.close()
    print("[DB] Database initialized successfully at:", config.DATABASE_PATH)

def seed_initial_data(conn):
    cursor = conn.cursor()
    # Seed default Admin and Seller accounts if not exist
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        admin_pass = generate_password_hash('admin123')
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ('admin', 'admin@example.com', admin_pass, 'admin')
        )
        print("[DB] Created default admin account: admin / admin123")

    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'seller'")
    if cursor.fetchone()[0] == 0:
        seller_pass = generate_password_hash('seller123')
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ('seller', 'seller@example.com', seller_pass, 'seller')
        )
        print("[DB] Created default seller account: seller / seller123")

    # Seed default Marketing Strategies for segments
    strategies = [
        (
            'Champions',
            'VIP Loyalty & Premium Retention',
            'Reward highest spending and most frequent customers to maintain brand advocacy.',
            'Personalized Email / Dedicated Concierge / WhatsApp VIP',
            'Early Access (No heavy discounts needed)',
            'Provide exclusive early access to new product drops, VIP concierge support, and surprise gift-with-purchase.'
        ),
        (
            'Loyal Customers',
            'Cross-Sell & Brand Advocacy',
            'Increase basket size and encourage customer referrals.',
            'Email Newsletter / App Push Notifications',
            '10% - 15% Bundle Discounts',
            'Offer tiered loyalty points, personalized product recommendations based on past purchase history, and referral incentives.'
        ),
        (
            'Potential Loyalists',
            'Nurture & Upsell Campaign',
            'Convert recent or promising buyers into frequent, habitual repeat shoppers.',
            'Targeted Email Sequences / Retargeting Ads',
            '15% Next-Order Incentive',
            'Share best-seller guides, customer reviews, and a time-limited discount for their second and third orders.'
        ),
        (
            'At-Risk Customers',
            'Win-Back & Churn Prevention',
            'Reactivate customers who used to buy frequently but have not visited recently.',
            'Urgent Win-Back Email / SMS Alert',
            '20% - 25% Comeback Special Discount',
            'Send \"We Miss You\" personalized messages, highlight popular new arrivals since their last visit, and offer an aggressive limited-time discount.'
        ),
        (
            'Lost / Inactive Customers',
            'Re-Engagement or Sunset Campaign',
            'Make a final attempt to revive completely lapsed customers before reducing ad spend on them.',
            'Low-cost Broadcast Email / Seasonal Clearance Banner',
            '30% Clearance Clearance Voucher',
            'Deliver seasonal clearance notifications or run an exit feedback survey to understand why they stopped purchasing.'
        )
    ]

    for strat in strategies:
        cursor.execute(
            """INSERT OR IGNORE INTO marketing_strategies 
               (segment_label, strategy_title, objective, recommended_channel, suggested_discount, action_plan)
               VALUES (?, ?, ?, ?, ?, ?)""",
            strat
        )
    conn.commit()

if __name__ == '__main__':
    init_db()
