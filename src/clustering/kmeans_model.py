import os
import pickle
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

class RFMClusterEngine:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mean_ = None
        self.std_ = None
        self.centroids_ = None
        self.k_ = None
        self.pca_components_ = None

    def fit_scaler(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0.0] = 1.0
        return (X - self.mean_) / self.std_

    def transform_scaler(self, X):
        return (X - self.mean_) / self.std_

    def compute_pca_2d(self, X_scaled):
        # Principal Component Analysis via Singular Value Decomposition (SVD)
        X_centered = X_scaled - np.mean(X_scaled, axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.pca_components_ = Vt[:2]
        return np.dot(X_centered, self.pca_components_.T)

    def compute_silhouette(self, X_scaled, labels, sample_size=500):
        # Fast representative silhouette evaluation
        n_samples = len(X_scaled)
        if n_samples <= sample_size:
            idx = np.arange(n_samples)
        else:
            np.random.seed(self.random_state)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            
        X_sub = X_scaled[idx]
        labels_sub = labels[idx]
        unique_labels = np.unique(labels_sub)
        
        if len(unique_labels) <= 1:
            return 0.0
            
        sils = []
        for i, pt in enumerate(X_sub):
            curr_label = labels_sub[i]
            same_mask = (labels_sub == curr_label)
            if np.sum(same_mask) <= 1:
                sils.append(0.0)
                continue
            # a_i = mean intra-cluster distance
            a_i = np.mean(np.linalg.norm(X_sub[same_mask] - pt, axis=1))
            
            # b_i = lowest mean inter-cluster distance
            b_i = np.inf
            for other_label in unique_labels:
                if other_label == curr_label:
                    continue
                other_mask = (labels_sub == other_label)
                if np.sum(other_mask) > 0:
                    d = np.mean(np.linalg.norm(X_sub[other_mask] - pt, axis=1))
                    if d < b_i:
                        b_i = d
                        
            denom = max(a_i, b_i)
            sils.append((b_i - a_i) / denom if denom > 0 else 0.0)
            
        return round(float(np.mean(sils)), 4)

    def evaluate_k(self, rfm_df, k_range=range(2, 9)):
        features = ['recency_log', 'frequency_log', 'monetary_log']
        X = rfm_df[features].values
        X_scaled = self.fit_scaler(X)
        
        k_list = []
        inertias = []
        silhouettes = []
        
        for k in k_range:
            np.random.seed(self.random_state)
            centroids, labels = kmeans2(X_scaled, k, minit='++', iter=30)
            
            # Compute WCSS / Inertia
            wcss = 0.0
            for c in range(k):
                pts = X_scaled[labels == c]
                if len(pts) > 0:
                    wcss += np.sum((pts - centroids[c]) ** 2)
                    
            sil = self.compute_silhouette(X_scaled, labels)
            
            k_list.append(int(k))
            inertias.append(round(float(wcss), 2))
            silhouettes.append(sil)
            
        best_idx = int(np.argmax(silhouettes))
        return {
            'k_values': k_list,
            'inertias': inertias,
            'silhouette_scores': silhouettes,
            'suggested_k': k_list[best_idx]
        }

    def fit_predict(self, rfm_df, k=4, model_save_path=None):
        features = ['recency_log', 'frequency_log', 'monetary_log']
        X = rfm_df[features].values
        X_scaled = self.fit_scaler(X)
        
        np.random.seed(self.random_state)
        centroids, labels = kmeans2(X_scaled, k, minit='++', iter=50)
        self.centroids_ = centroids
        self.k_ = k
        
        pca_coords = self.compute_pca_2d(X_scaled)
        
        # Inertia & Silhouette
        wcss = 0.0
        for c in range(k):
            pts = X_scaled[labels == c]
            if len(pts) > 0:
                wcss += np.sum((pts - centroids[c]) ** 2)
                
        sil_score = self.compute_silhouette(X_scaled, labels, sample_size=1000)
        
        result_df = rfm_df.copy()
        result_df['cluster_id'] = labels
        result_df['pca_x'] = np.round(pca_coords[:, 0], 3)
        result_df['pca_y'] = np.round(pca_coords[:, 1], 3)
        
        # Summary profiles on original RFM scales
        profiles = result_df.groupby('cluster_id').agg(
            customer_count=('customer_id', 'count'),
            mean_recency=('recency', 'mean'),
            median_recency=('recency', 'median'),
            mean_frequency=('frequency', 'mean'),
            median_frequency=('frequency', 'median'),
            mean_monetary=('monetary', 'mean'),
            median_monetary=('monetary', 'median'),
            total_revenue=('monetary', 'sum')
        ).round(2).reset_index()
        
        # Save model artifact
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            with open(model_save_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"[MODEL] Saved engine to {model_save_path}")
            
        return result_df, profiles, sil_score, round(float(wcss), 2)

if __name__ == '__main__':
    rfm_path = r'C:\customer-segmentation-system\data\processed\customer_rfm.csv'
    if os.path.exists(rfm_path):
        print("[TEST] Running RFMClusterEngine on customer_rfm.csv...")
        rfm_df = pd.read_csv(rfm_path)
        engine = RFMClusterEngine(random_state=42)
        eval_res = engine.evaluate_k(rfm_df)
        print("[RESULTS] Hyperparameter Evaluation:")
        for k, inertia, sil in zip(eval_res['k_values'], eval_res['inertias'], eval_res['silhouette_scores']):
            print(f"  - k={k}: Inertia={inertia:,.2f}, Silhouette Score={sil:.4f}")
            
        model_path = r'C:\customer-segmentation-system\models\kmeans_model.pkl'
        clustered_df, profiles, sil, inertia = engine.fit_predict(rfm_df, k=4, model_save_path=model_path)
        print(f"\n[FITTED] k=4 -> Silhouette: {sil}, Inertia: {inertia:,.2f}")
        print("\n[PROFILES] Cluster Behavioral Profiles:")
        print(profiles)
