from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    """Enhanced anomaly detection class with multiple advanced techniques"""
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def enhanced_feature_engineering(self, df):
        """Create comprehensive feature set for better anomaly detection"""
        df_enhanced = df.copy()
        
        # Basic derived features
        df_enhanced['speed_category'] = pd.cut(df['pc_speed'], 
                                             bins=[0, 30, 60, 100], 
                                             labels=['Low', 'Medium', 'High'])
        df_enhanced['speed_category_encoded'] = df_enhanced['speed_category'].cat.codes
        
        # Advanced behavior features
        df_enhanced['aggressive_steering'] = (np.abs(df['pc_steering']) > df['pc_steering'].quantile(0.9)).astype(int)
        df_enhanced['hard_braking'] = (df['pc_brake'] > df['pc_brake'].quantile(0.9)).astype(int)
        df_enhanced['speed_steering_ratio'] = df['pc_speed'] / (np.abs(df['pc_steering']) + 1)
        df_enhanced['brake_intensity'] = df['pc_brake'] * df['pc_speed']
        
        # Interaction features
        df_enhanced['speed_brake_interaction'] = df['pc_speed'] * df['pc_brake']
        df_enhanced['steering_brake_interaction'] = np.abs(df['pc_steering']) * df['pc_brake']
        df_enhanced['combined_intensity'] = np.sqrt(df['pc_speed']**2 + df['pc_steering']**2 + df['pc_brake']**2)
        
        # Statistical features (rolling windows)
        window_sizes = [3, 5, 10]
        for window in window_sizes:
            if len(df) > window:
                df_enhanced[f'speed_rolling_mean_{window}'] = df['pc_speed'].rolling(window=window, min_periods=1).mean()
                df_enhanced[f'speed_rolling_std_{window}'] = df['pc_speed'].rolling(window=window, min_periods=1).std()
                df_enhanced[f'steering_rolling_var_{window}'] = df['pc_steering'].rolling(window=window, min_periods=1).var()
                df_enhanced[f'brake_rolling_max_{window}'] = df['pc_brake'].rolling(window=window, min_periods=1).max()
        
        # Z-scores and percentile ranks
        for col in ['pc_speed', 'pc_steering', 'pc_brake']:
            df_enhanced[f'{col}_zscore'] = np.abs(stats.zscore(df[col]))
            df_enhanced[f'{col}_percentile'] = df[col].rank(pct=True)
        
        # Lag features
        for lag in [1, 2, 3]:
            if len(df) > lag:
                df_enhanced[f'speed_lag_{lag}'] = df['pc_speed'].shift(lag)
                df_enhanced[f'steering_lag_{lag}'] = df['pc_steering'].shift(lag)
                df_enhanced[f'brake_lag_{lag}'] = df['pc_brake'].shift(lag)
        
        # Change features (derivatives)
        df_enhanced['speed_change'] = df['pc_speed'].diff()
        df_enhanced['steering_change'] = df['pc_steering'].diff()
        df_enhanced['brake_change'] = df['pc_brake'].diff()
        
        # Smooth features using exponential moving average
        alpha = 0.3
        df_enhanced['speed_ema'] = df['pc_speed'].ewm(alpha=alpha).mean()
        df_enhanced['steering_ema'] = df['pc_steering'].ewm(alpha=alpha).mean()
        df_enhanced['brake_ema'] = df['pc_brake'].ewm(alpha=alpha).mean()
        
        return df_enhanced
    
    def mahalanobis_distance_detection(self, X):
        """Calculate Mahalanobis distance for outlier detection"""
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(X.T)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            mean_vector = np.mean(X, axis=0)
            
            # Calculate Mahalanobis distance for each point
            distances = []
            for i in range(X.shape[0]):
                diff = X[i] - mean_vector
                distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                distances.append(distance)
            
            distances = np.array(distances)
            threshold = np.percentile(distances, (1 - self.contamination) * 100)
            predictions = np.where(distances > threshold, -1, 1)
            
            return predictions, distances
        except:
            # Fallback to simple distance if Mahalanobis fails
            distances = np.sqrt(np.sum((X - np.mean(X, axis=0))**2, axis=1))
            threshold = np.percentile(distances, (1 - self.contamination) * 100)
            predictions = np.where(distances > threshold, -1, 1)
            return predictions, distances
    
    def dbscan_anomaly_detection(self, X):
        """Use DBSCAN clustering to identify anomalies"""
        # Tune DBSCAN parameters
        eps_values = np.linspace(0.1, 2.0, 10)
        min_samples_values = [3, 5, 10]
        
        best_score = -1
        best_params = {}
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                    if len(set(labels)) > 1:  # At least one cluster found
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                except:
                    continue
        
        if best_params:
            dbscan = DBSCAN(**best_params)
            labels = dbscan.fit_predict(X)
            # Points labeled as -1 are anomalies in DBSCAN
            predictions = np.where(labels == -1, -1, 1)
        else:
            # Fallback: label small percentage as anomalies
            n_anomalies = int(self.contamination * len(X))
            predictions = np.ones(len(X))
            predictions[:n_anomalies] = -1
        
        return predictions, labels if best_params else None
    
    def ensemble_with_weights(self, X):
        """Advanced ensemble with weighted voting based on model performance"""
        print("🤖 Training advanced ensemble with weighted voting...")
        
        models_results = {}
        
        # 1. Isolation Forest with tuning
        iso_forest = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state, 
            n_estimators=300,
            max_features=0.8,
            bootstrap=True
        )
        iso_predictions = iso_forest.fit_predict(X)
        iso_scores = iso_forest.decision_function(X)
        models_results['isolation_forest'] = {
            'predictions': iso_predictions,
            'scores': iso_scores,
            'model': iso_forest
        }
        
        # 2. One-Class SVM with different kernels
        for kernel in ['rbf', 'poly']:
            svm_model = OneClassSVM(
                gamma='scale', 
                nu=self.contamination,
                kernel=kernel
            )
            svm_predictions = svm_model.fit_predict(X)
            models_results[f'one_class_svm_{kernel}'] = {
                'predictions': svm_predictions,
                'scores': svm_model.decision_function(X),
                'model': svm_model
            }
        
        # 3. Local Outlier Factor with different neighbors
        for n_neighbors in [10, 20, 30]:
            lof_model = LocalOutlierFactor(
                contamination=self.contamination, 
                n_neighbors=n_neighbors
            )
            lof_predictions = lof_model.fit_predict(X)
            models_results[f'lof_{n_neighbors}'] = {
                'predictions': lof_predictions,
                'scores': lof_model.negative_outlier_factor_,
                'model': lof_model
            }
        
        # 4. Mahalanobis distance
        mahal_predictions, mahal_distances = self.mahalanobis_distance_detection(X)
        models_results['mahalanobis'] = {
            'predictions': mahal_predictions,
            'scores': -mahal_distances,  # Negative for consistency
            'model': None
        }
        
        # 5. DBSCAN-based detection
        dbscan_predictions, dbscan_labels = self.dbscan_anomaly_detection(X)
        models_results['dbscan'] = {
            'predictions': dbscan_predictions,
            'scores': None,
            'model': None
        }
        
        # Calculate weights based on agreement with other models
        model_names = list(models_results.keys())
        weights = {}
        
        for model_name in model_names:
            agreement_scores = []
            for other_model in model_names:
                if model_name != other_model:
                    agreement = np.mean(
                        models_results[model_name]['predictions'] == 
                        models_results[other_model]['predictions']
                    )
                    agreement_scores.append(agreement)
            weights[model_name] = np.mean(agreement_scores)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"Model weights: {weights}")
        
        # Weighted ensemble voting
        weighted_votes = np.zeros(len(X))
        for model_name, weight in weights.items():
            predictions = models_results[model_name]['predictions']
            weighted_votes += weight * predictions
        
        # Final predictions based on weighted majority
        ensemble_predictions = np.where(weighted_votes < 0, -1, 1)
        
        # Store models and results
        self.models = {name: result['model'] for name, result in models_results.items() 
                      if result['model'] is not None}
        
        return {
            'ensemble': ensemble_predictions,
            'models_results': models_results,
            'weights': weights,
            'weighted_votes': weighted_votes
        }
    
    def dimensionality_reduction_analysis(self, X, df):
        """Use PCA for visualization and feature importance analysis"""
        # Apply PCA
        pca = PCA(n_components=min(3, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # Calculate feature importance based on PCA components
        feature_importance = np.abs(pca.components_).mean(axis=0)
        
        # Normalize feature importance
        feature_importance = feature_importance / feature_importance.sum()
        
        return X_pca, pca, feature_importance
    
    def advanced_preprocessing(self, df, feature_columns):
        """Advanced preprocessing with multiple scaling options"""
        df_clean = df[feature_columns].copy()
        
        # Handle missing values with multiple strategies
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # For numeric columns, use median
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
            df_clean[numeric_columns].median()
        )
        
        # Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())
        
        # Remove extreme outliers using IQR method
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        # Apply multiple scaling techniques and combine
        scalers = {
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'standard': StandardScaler()
        }
        
        scaled_data = {}
        for name, scaler in scalers.items():
            scaled_data[name] = scaler.fit_transform(df_clean)
            self.scalers[name] = scaler
        
        # Use robust scaler as primary, but keep others for ensemble
        primary_scaled = scaled_data['robust']
        
        return primary_scaled, df_clean.index, scaled_data
    
    def comprehensive_evaluation(self, results, df_valid):
        """Comprehensive evaluation with multiple metrics"""
        print("\n📊 Comprehensive Model Evaluation:")
        print("=" * 60)
        
        ensemble_predictions = results['ensemble']
        models_results = results['models_results']
        
        # Overall statistics
        n_anomalies = np.sum(ensemble_predictions == -1)
        n_normal = np.sum(ensemble_predictions == 1)
        anomaly_rate = n_anomalies / len(ensemble_predictions) * 100
        
        print(f"📈 Overall Statistics:")
        print(f"  Total samples: {len(ensemble_predictions)}")
        print(f"  Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")
        print(f"  Normal samples: {n_normal} ({100-anomaly_rate:.2f}%)")
        
        # Model agreement analysis
        print(f"\n🤝 Model Agreement Analysis:")
        model_predictions = np.column_stack([
            models_results[name]['predictions'] for name in models_results.keys()
        ])
        
        # Calculate pairwise agreement
        n_models = model_predictions.shape[1]
        agreements = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                agreement = np.mean(model_predictions[:, i] == model_predictions[:, j])
                agreements.append(agreement)
        
        print(f"  Average pairwise agreement: {np.mean(agreements):.3f}")
        print(f"  Agreement range: {np.min(agreements):.3f} - {np.max(agreements):.3f}")
        
        # Anomaly score analysis
        print(f"\n📊 Anomaly Score Analysis:")
        if 'iso_scores' in [r for r in models_results.values() if r['scores'] is not None]:
            iso_scores = models_results['isolation_forest']['scores']
            anomaly_scores = iso_scores[ensemble_predictions == -1]
            normal_scores = iso_scores[ensemble_predictions == 1]
            
            print(f"  Anomaly scores - Mean: {np.mean(anomaly_scores):.3f}, Std: {np.std(anomaly_scores):.3f}")
            print(f"  Normal scores - Mean: {np.mean(normal_scores):.3f}, Std: {np.std(normal_scores):.3f}")
            print(f"  Score separation: {np.mean(normal_scores) - np.mean(anomaly_scores):.3f}")
        
        return {
            'anomaly_rate': anomaly_rate,
            'model_agreement': np.mean(agreements),
            'n_anomalies': n_anomalies,
            'n_normal': n_normal
        }
    
    def advanced_visualization(self, df, predictions, results, save_path="enhanced_anomaly_analysis.png"):
        """Create advanced visualizations with multiple perspectives"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Color mapping
        colors = ['red' if x == -1 else 'blue' for x in predictions]
        
        # 1. Speed vs Steering (traditional)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(df['pc_speed'], df['pc_steering'], c=colors, alpha=0.6, s=30)
        ax1.set_xlabel('Speed')
        ax1.set_ylabel('Steering')
        ax1.set_title('Speed vs Steering')
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed vs Brake (traditional)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df['pc_speed'], df['pc_brake'], c=colors, alpha=0.6, s=30)
        ax2.set_xlabel('Speed')
        ax2.set_ylabel('Brake')
        ax2.set_title('Speed vs Brake')
        ax2.grid(True, alpha=0.3)
        
        # 3. 3D scatter plot
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.scatter(df['pc_speed'], df['pc_steering'], df['pc_brake'], c=colors, alpha=0.6, s=20)
        ax3.set_xlabel('Speed')
        ax3.set_ylabel('Steering')
        ax3.set_zlabel('Brake')
        ax3.set_title('3D Feature Space')
        
        # 4. Model agreement heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        models_results = results['models_results']
        model_names = list(models_results.keys())
        agreement_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    agreement = np.mean(
                        models_results[model1]['predictions'] == 
                        models_results[model2]['predictions']
                    )
                    agreement_matrix[i, j] = agreement
                else:
                    agreement_matrix[i, j] = 1.0
        
        sns.heatmap(agreement_matrix, 
                   xticklabels=[name[:8] for name in model_names],
                   yticklabels=[name[:8] for name in model_names],
                   annot=True, fmt='.2f', ax=ax4, cmap='viridis')
        ax4.set_title('Model Agreement Matrix')
        
        # 5. Time series with anomalies
        ax5 = fig.add_subplot(gs[1, :2])
        ax5.plot(df.index, df['pc_speed'], alpha=0.7, label='Speed', color='blue')
        ax5.plot(df.index, df['pc_steering'], alpha=0.7, label='Steering', color='green')
        ax5.plot(df.index, df['pc_brake'] * 50, alpha=0.7, label='Brake (x50)', color='orange')
        
        anomaly_indices = df.index[predictions == -1]
        ax5.scatter(anomaly_indices, df.loc[anomaly_indices, 'pc_speed'], 
                   color='red', s=50, label='Anomalies', zorder=5)
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Values')
        ax5.set_title('Time Series with Anomalies')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Anomaly score distribution
        ax6 = fig.add_subplot(gs[1, 2])
        if 'isolation_forest' in models_results and models_results['isolation_forest']['scores'] is not None:
            scores = models_results['isolation_forest']['scores']
            ax6.hist(scores[predictions == 1], bins=30, alpha=0.7, label='Normal', color='blue')
            ax6.hist(scores[predictions == -1], bins=30, alpha=0.7, label='Anomaly', color='red')
            ax6.set_xlabel('Isolation Forest Score')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Score Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Model weights visualization
        ax7 = fig.add_subplot(gs[1, 3])
        weights = results['weights']
        model_names_short = [name[:10] for name in weights.keys()]
        weight_values = list(weights.values())
        
        bars = ax7.bar(range(len(weights)), weight_values, color='skyblue')
        ax7.set_xlabel('Models')
        ax7.set_ylabel('Weight')
        ax7.set_title('Ensemble Model Weights')
        ax7.set_xticks(range(len(weights)))
        ax7.set_xticklabels(model_names_short, rotation=45, ha='right')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, weight_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 8. Feature correlation with anomalies
        ax8 = fig.add_subplot(gs[2, :2])
        feature_cols = ['pc_speed', 'pc_steering', 'pc_brake']
        correlations = []
        
        for col in feature_cols:
            corr = np.corrcoef(df[col], (predictions == -1).astype(int))[0, 1]
            correlations.append(abs(corr))
        
        bars = ax8.bar(feature_cols, correlations, color=['lightcoral', 'lightgreen', 'lightblue'])
        ax8.set_ylabel('Absolute Correlation with Anomalies')
        ax8.set_title('Feature-Anomaly Correlations')
        ax8.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, correlations):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 9. Box plot of features by anomaly status
        ax9 = fig.add_subplot(gs[2, 2:])
        data_for_box = []
        labels_for_box = []
        
        for col in feature_cols:
            data_for_box.extend([df[col][predictions == 1], df[col][predictions == -1]])
            labels_for_box.extend([f'{col}\nNormal', f'{col}\nAnomaly'])
        
        box_plot = ax9.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # Color the boxes
        colors_box = ['lightblue', 'lightcoral'] * len(feature_cols)
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
        
        ax9.set_ylabel('Feature Values')
        ax9.set_title('Feature Distributions by Anomaly Status')
        ax9.grid(True, alpha=0.3)
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Enhanced Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Enhanced visualization saved to {save_path}")

def main():
    """Main execution function with enhanced anomaly detection"""
    print("🚗 Advanced Driving Anomaly Detection System v2.0")
    print("=" * 60)
    
    # Initialize detector
    detector = AdvancedAnomalyDetector(contamination=0.05, random_state=42)
    
    # Load data
    print("📊 Loading and exploring data...")
    df = pd.read_csv("realistic_driving_data.csv")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Enhanced feature engineering
    print("\n🔧 Creating enhanced features...")
    df_enhanced = detector.enhanced_feature_engineering(df)
    
    # Select features intelligently
    numeric_features = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target-like columns if any
    feature_columns = [col for col in numeric_features if 'anomaly' not in col.lower()]
    
    print(f"Total features created: {len(feature_columns)}")
    print(f"Selected features: {feature_columns[:10]}...")  # Show first 10
    
    # Advanced preprocessing
    print("\n⚙️ Advanced preprocessing...")
    X_scaled, valid_indices, scaled_data = detector.advanced_preprocessing(df_enhanced, feature_columns)
    df_valid = df_enhanced.loc[valid_indices].reset_index(drop=True)
    
    print(f"Samples after preprocessing: {len(df_valid)}")
    
    # Dimensionality analysis
    print("\n📐 Dimensionality reduction analysis...")
    X_pca, pca, feature_importance = detector.dimensionality_reduction_analysis(X_scaled, df_valid)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance explained: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Train advanced ensemble
    print("\n🤖 Training advanced ensemble...")
    results = detector.ensemble_with_weights(X_scaled)
    
    # Comprehensive evaluation
    evaluation_results = detector.comprehensive_evaluation(results, df_valid)
    
    # Prepare final output
    final_predictions = results['ensemble']
    df_valid['anomaly'] = final_predictions
    df_valid['anomaly_label'] = df_valid['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    # Add scores from different models
    models_results = results['models_results']
    if 'isolation_forest' in models_results:
        df_valid['iso_score'] = models_results['isolation_forest']['scores']
    if 'mahalanobis' in models_results:
        df_valid['mahalanobis_score'] = models_results['mahalanobis']['scores']
    
    df_valid['ensemble_vote'] = results['weighted_votes']
    
    # Save comprehensive results
    output_columns = ['pc_speed', 'pc_steering', 'pc_brake', 'anomaly_label', 
                     'ensemble_vote'] + [col for col in df_valid.columns if '_score' in col]
    output_columns = [col for col in output_columns if col in df_valid.columns]
    
    df_valid[output_columns].to_csv("enhanced_anomaly_results.csv", index=False)
    
    # Create advanced visualizations
    print("\n📊 Creating advanced visualizations...")
    detector.advanced_visualization(df_valid, final_predictions, results)
    
    # Final summary
    print("\n✅ Enhanced Anomaly Detection Complete!")
    print("=" * 60)
    print(f"📊 Results Summary:")
    print(f"  • Total samples processed: {len(df_valid)}")
    print(f"  • Anomalies detected: {evaluation_results['n_anomalies']} ({evaluation_results['anomaly_rate']:.2f}%)")
    print(f"  • Model agreement score: {evaluation_results['model_agreement']:.3f}")
    print(f"  • Features engineered: {len(feature_columns)}")
    
    # Show top anomalies
    print(f"\n🚨 Top 5 Most Anomalous Samples:")
    anomaly_indices = df_valid[df_valid['anomaly'] == -1].index
    if len(anomaly_indices) > 0:
        if 'iso_score' in df_valid.columns:
            top_anomalies = df_valid.loc[anomaly_indices].nsmallest(5, 'iso_score')
            cols_to_show = ['pc_speed', 'pc_steering', 'pc_brake', 'iso_score', 'ensemble_vote']
        else:
            top_anomalies = df_valid.loc[anomaly_indices].nsmallest(5, 'ensemble_vote')
            cols_to_show = ['pc_speed', 'pc_steering', 'pc_brake', 'ensemble_vote']
        
        cols_to_show = [col for col in cols_to_show if col in top_anomalies.columns]
        print(top_anomalies[cols_to_show].to_string())
    
    print(f"\n📁 Files Generated:")
    print(f"  • enhanced_anomaly_results.csv (detailed results)")
    print(f"  • enhanced_anomaly_analysis.png (comprehensive visualizations)")
    
    return detector, results, df_valid

if __name__ == "__main__":
    detector, results, df_valid = main()
