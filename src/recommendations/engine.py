import os
import pandas as pd
import numpy as np

MARKETING_PLAYBOOKS = {
    'Champions': {
        'badge_color': 'success',
        'title': 'VIP Loyalty & Premium Retention',
        'objective': 'Reward high-spending brand champions and maintain lifetime loyalty.',
        'channels': ['Personalized Email', 'VIP WhatsApp Concierge', 'Exclusive Member Events'],
        'discount': 'Early Access / Free Premium Gifts (Avoid heavy price discounts)',
        'action_plan': [
            'Enroll automatically in VIP Tier with dedicated customer support.',
            'Provide 48-hour early preview access to new seasonal collections.',
            'Send surprise anniversary and birthday celebration rewards.',
            'Invite to beta-test new products and share feedback.'
        ]
    },
    'Loyal Customers': {
        'badge_color': 'primary',
        'title': 'Cross-Sell & Basket Size Growth',
        'objective': 'Increase average order value and incentivize customer referrals.',
        'channels': ['Email Newsletters', 'Mobile App Push Notifications'],
        'discount': '10% - 15% Tiered Bundle Discounts',
        'action_plan': [
            'Offer tiered threshold discounts (e.g., Save  when you spend ).',
            'Cross-sell complementary products based on past category purchases.',
            'Launch referral rewards: give , get  for referring a friend.'
        ]
    },
    'Potential Loyalists': {
        'badge_color': 'info',
        'title': 'Nurture & Repeat Purchase Accelerator',
        'objective': 'Convert promising recent buyers into frequent, habitual shoppers.',
        'channels': ['Automated Email Sequences', 'Retargeting Social Ads'],
        'discount': '15% Off Next Purchase (Expiring in 14 days)',
        'action_plan': [
            'Send educational onboarding emails highlighting brand guarantees and best practices.',
            'Offer a time-sensitive second-order incentive voucher.',
            'Highlight top 5 trending products with high 5-star customer reviews.'
        ]
    },
    'At-Risk Customers': {
        'badge_color': 'warning',
        'title': 'Win-Back & Churn Prevention',
        'objective': 'Reactivate formerly frequent shoppers before they lapse permanently.',
        'channels': ['High-Priority SMS', 'Personalized Win-Back Email'],
        'discount': '20% - 25% Limited-Time Comeback Offer',
        'action_plan': [
            'Send emotional win-back message: \"We miss you! Here is 20% off your next visit\".',
            'Showcase popular new arrivals and improvements introduced since their last visit.',
            'Trigger an automated SMS reminder 48 hours before coupon expiration.'
        ]
    },
    'Lost / Inactive Customers': {
        'badge_color': 'danger',
        'title': 'Re-Engagement & Sunset Protocol',
        'objective': 'Make a final cost-effective reactivation attempt before suppressing marketing spend.',
        'channels': ['Low-Cost Batch Email', 'Seasonal Clearance Blast'],
        'discount': '30% Clearance Voucher',
        'action_plan': [
            'Include in broad seasonal clearance and Black Friday email blasts.',
            'Deploy a 1-question exit survey: \"How can we improve our service for you?\".',
            'Unsubscribe non-responsive contacts after 90 days to protect email domain deliverability.'
        ]
    }
}

def assign_segment_labels(profiles_df):
    """
    Dynamically maps cluster IDs to human-readable business segment labels
    by evaluating their actual RFM centroids.
    """
    profiles = profiles_df.copy()
    k = len(profiles)
    
    # Sort clusters by monetary spend descending
    profiles = profiles.sort_values(by='mean_monetary', ascending=False).reset_index(drop=True)
    
    cluster_mapping = {}
    
    if k == 2:
        cluster_mapping[profiles.loc[0, 'cluster_id']] = 'Champions'
        cluster_mapping[profiles.loc[1, 'cluster_id']] = 'Lost / Inactive Customers'
    elif k == 3:
        cluster_mapping[profiles.loc[0, 'cluster_id']] = 'Champions'
        cluster_mapping[profiles.loc[1, 'cluster_id']] = 'Loyal Customers'
        cluster_mapping[profiles.loc[2, 'cluster_id']] = 'At-Risk Customers'
    elif k == 4:
        cluster_mapping[profiles.loc[0, 'cluster_id']] = 'Champions'
        # Distinguish between Loyal vs At-Risk vs Lost among the rest
        rest = profiles.iloc[1:].copy()
        rest_by_recency = rest.sort_values(by='mean_recency', ascending=True).reset_index(drop=True)
        cluster_mapping[rest_by_recency.loc[0, 'cluster_id']] = 'Potential Loyalists'
        cluster_mapping[rest_by_recency.loc[1, 'cluster_id']] = 'Loyal Customers'
        cluster_mapping[rest_by_recency.loc[2, 'cluster_id']] = 'Lost / Inactive Customers'
    else:
        # k >= 5
        cluster_mapping[profiles.loc[0, 'cluster_id']] = 'Champions'
        rest = profiles.iloc[1:].copy()
        rest_by_rec = rest.sort_values(by='mean_recency', ascending=True).reset_index(drop=True)
        cluster_mapping[rest_by_rec.loc[0, 'cluster_id']] = 'Potential Loyalists'
        cluster_mapping[rest_by_rec.loc[1, 'cluster_id']] = 'Loyal Customers'
        cluster_mapping[rest_by_rec.loc[-1, 'cluster_id']] = 'Lost / Inactive Customers'
        # Any remaining become At-Risk
        for cid in profiles['cluster_id']:
            if cid not in cluster_mapping:
                cluster_mapping[cid] = 'At-Risk Customers'
                
    return cluster_mapping

def apply_recommendations_to_customers(clustered_df, cluster_mapping):
    df = clustered_df.copy()
    df['segment_label'] = df['cluster_id'].map(cluster_mapping)
    
    # Attach marketing meta
    df['badge_color'] = df['segment_label'].apply(lambda x: MARKETING_PLAYBOOKS.get(x, {}).get('badge_color', 'secondary'))
    df['marketing_title'] = df['segment_label'].apply(lambda x: MARKETING_PLAYBOOKS.get(x, {}).get('title', 'Standard Outreach'))
    
    return df

def get_segment_strategy_summary(cluster_mapping, profiles_df):
    summary = []
    for _, row in profiles_df.iterrows():
        cid = int(row['cluster_id'])
        label = cluster_mapping.get(cid, f'Segment {cid}')
        playbook = MARKETING_PLAYBOOKS.get(label, {})
        summary.append({
            'cluster_id': cid,
            'segment_label': label,
            'badge_color': playbook.get('badge_color', 'secondary'),
            'customer_count': int(row['customer_count']),
            'total_revenue': float(row['total_revenue']),
            'mean_recency': float(row['mean_recency']),
            'mean_frequency': float(row['mean_frequency']),
            'mean_monetary': float(row['mean_monetary']),
            'strategy_title': playbook.get('title', ''),
            'objective': playbook.get('objective', ''),
            'channels': playbook.get('channels', []),
            'suggested_discount': playbook.get('discount', ''),
            'action_plan': playbook.get('action_plan', [])
        })
    return summary

if __name__ == '__main__':
    from src.clustering.kmeans_model import RFMClusterEngine
    rfm_path = r'C:\customer-segmentation-system\data\processed\customer_rfm.csv'
    if os.path.exists(rfm_path):
        print("[TEST] Running Recommendation Engine...")
        rfm_df = pd.read_csv(rfm_path)
        engine = RFMClusterEngine(random_state=42)
        clustered_df, profiles, sil, inertia = engine.fit_predict(rfm_df, k=4)
        mapping = assign_segment_labels(profiles)
        print("\n[MAPPING] Cluster ID to Business Segments:")
        for cid, label in mapping.items():
            print(f"  - Cluster {cid} -> {label}")
        
        final_df = apply_recommendations_to_customers(clustered_df, mapping)
        out_final = r'C:\customer-segmentation-system\data\processed\final_customer_segments.csv'
        final_df.to_csv(out_final, index=False)
        print(f"\n[SAVED] Enriched customer segments saved to: {out_final}")
        
        strategies = get_segment_strategy_summary(mapping, profiles)
        print("\n[SAMPLE STRATEGY] First Segment Action Plan:")
        print(f"  Segment: {strategies[0]['segment_label']}")
        print(f"  Title: {strategies[0]['strategy_title']}")
        print(f"  Discount: {strategies[0]['suggested_discount']}")
