from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(filename):
    """Load data and perform basic exploration"""
    df = pd.read_csv(filename)
    print("📊 Data Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n📈 Statistical Summary:")
    print(df.describe())
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    return df

def feature_engineering(df):
    """Create additional features for better anomaly detection"""
    df_enhanced = df.copy()
    
    # Speed-related features
    df_enhanced['speed_category'] = pd.cut(df['pc_speed'], 
                                         bins=[0, 30, 60, 100], 
                                         labels=['Low', 'Medium', 'High'])
    df_enhanced['speed_category_encoded'] = df_enhanced['speed_category'].cat.codes
    
    # Driving behavior features
    df_enhanced['aggressive_steering'] = np.abs(df['pc_steering']) > df['pc_steering'].quantile(0.9)
    df_enhanced['hard_braking'] = df['pc_brake'] > df['pc_brake'].quantile(0.9)
    df_enhanced['speed_steering_ratio'] = df['pc_speed'] / (np.abs(df['pc_steering']) + 1)
    df_enhanced['brake_intensity'] = df['pc_brake'] * df['pc_speed']
    
    # Statistical features (rolling windows)
    window_size = min(5, len(df) // 10)  # Adaptive window size
    if window_size > 1:
        df_enhanced['speed_rolling_std'] = df['pc_speed'].rolling(window=window_size, min_periods=1).std()
        df_enhanced['steering_rolling_mean'] = df['pc_steering'].rolling(window=window_size, min_periods=1).mean()
        df_enhanced['brake_rolling_max'] = df['pc_brake'].rolling(window=window_size, min_periods=1).max()
    
    # Z-scores for outlier detection
    for col in ['pc_speed', 'pc_steering', 'pc_brake']:
        df_enhanced[f'{col}_zscore'] = np.abs(stats.zscore(df[col]))
    
    return df_enhanced

def preprocess_features(df, feature_columns):
    """Preprocess features with scaling and outlier handling"""
    # Handle missing values
    df_clean = df[feature_columns].fillna(df[feature_columns].median())
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < 3]
    
    # Scale features
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    scaled_features = scaler.fit_transform(df_clean)
    
    return scaled_features, scaler, df_clean.index

def tune_isolation_forest(X, contamination_values=[0.01, 0.05, 0.1, 0.15]):
    """Tune Isolation Forest hyperparameters"""
    best_score = -np.inf
    best_params = {}
    best_model = None
    
    param_grid = {
        'contamination': contamination_values,
        'n_estimators': [100, 200],
        'max_features': [1.0, 0.8],
        'random_state': [42]
    }
    
    print("🔧 Tuning Isolation Forest...")
    for params in ParameterGrid(param_grid):
        model = IsolationForest(**params)
        model.fit(X)
        score = model.score_samples(X).mean()
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    
    print(f"✅ Best Isolation Forest params: {best_params}")
    return best_model, best_params

def ensemble_anomaly_detection(X, contamination=0.05):
    """Use ensemble of different anomaly detection algorithms"""
    print("🤖 Training ensemble of anomaly detectors...")
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso_predictions = iso_forest.fit_predict(X)
    
    # One-Class SVM
    svm_model = OneClassSVM(gamma='scale', nu=contamination)
    svm_predictions = svm_model.fit_predict(X)
    
    # Local Outlier Factor
    lof_model = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
    lof_predictions = lof_model.fit_predict(X)
    
    # Ensemble voting (majority vote)
    predictions_matrix = np.column_stack([iso_predictions, svm_predictions, lof_predictions])
    ensemble_predictions = np.apply_along_axis(lambda x: 1 if np.sum(x == 1) >= 2 else -1, 
                                             axis=1, arr=predictions_matrix)
    
    # Get anomaly scores
    iso_scores = iso_forest.decision_function(X)
    lof_scores = lof_model.negative_outlier_factor_
    
    return {
        'ensemble': ensemble_predictions,
        'isolation_forest': iso_predictions,
        'one_class_svm': svm_predictions,
        'local_outlier_factor': lof_predictions,
        'iso_scores': iso_scores,
        'lof_scores': lof_scores,
        'models': {
            'isolation_forest': iso_forest,
            'one_class_svm': svm_model,
            'local_outlier_factor': lof_model
        }
    }

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate anomaly detection performance"""
    print(f"\n📊 {model_name} Evaluation:")
    
    # Convert to binary format for evaluation
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    print(f"Total samples: {len(y_pred)}")
    print(f"Anomalies detected: {np.sum(y_pred == -1)} ({np.sum(y_pred == -1)/len(y_pred)*100:.2f}%)")
    print(f"Normal samples: {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.2f}%)")
    
    return y_pred_binary

def visualize_anomalies(df, predictions, scores=None, save_path="improved_anomaly_plot.png"):
    """Create comprehensive visualization of anomalies"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Improved Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # Color mapping
    colors = ['red' if x == -1 else 'blue' for x in predictions]
    labels = ['Anomaly' if x == -1 else 'Normal' for x in predictions]
    
    # Plot 1: Speed vs Steering
    axes[0, 0].scatter(df['pc_speed'], df['pc_steering'], c=colors, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Speed')
    axes[0, 0].set_ylabel('Steering')
    axes[0, 0].set_title('Speed vs Steering')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Speed vs Brake
    axes[0, 1].scatter(df['pc_speed'], df['pc_brake'], c=colors, alpha=0.6, s=30)
    axes[0, 1].set_xlabel('Speed')
    axes[0, 1].set_ylabel('Brake')
    axes[0, 1].set_title('Speed vs Brake')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time series view
    axes[1, 0].plot(df.index, df['pc_speed'], alpha=0.7, label='Speed')
    anomaly_indices = df.index[predictions == -1]
    axes[1, 0].scatter(anomaly_indices, df.loc[anomaly_indices, 'pc_speed'], 
                      color='red', s=50, label='Anomalies', zorder=5)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Speed')
    axes[1, 0].set_title('Speed Time Series with Anomalies')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Anomaly scores distribution (if available)
    if scores is not None:
        axes[1, 1].hist(scores[predictions == 1], bins=30, alpha=0.7, label='Normal', color='blue')
        axes[1, 1].hist(scores[predictions == -1], bins=30, alpha=0.7, label='Anomaly', color='red')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Anomaly Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Feature importance plot instead
        feature_names = ['Speed', 'Steering', 'Brake']
        feature_importance = [df['pc_speed'].std(), np.abs(df['pc_steering']).mean(), df['pc_brake'].std()]
        axes[1, 1].bar(feature_names, feature_importance, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 1].set_ylabel('Feature Variability')
        axes[1, 1].set_title('Feature Importance (Std/Mean)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Visualization saved to {save_path}")

def main():
    """Main execution function"""
    print("🚗 Advanced Driving Anomaly Detection System")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data("realistic_driving_data.csv")
    
    # Feature engineering
    df_enhanced = feature_engineering(df)
    
    # Select features for modeling
    original_features = ['pc_speed', 'pc_steering', 'pc_brake']
    enhanced_features = ['pc_speed', 'pc_steering', 'pc_brake', 'speed_category_encoded',
                        'aggressive_steering', 'hard_braking', 'speed_steering_ratio', 
                        'brake_intensity']
    
    # Add rolling features if they exist
    rolling_features = [col for col in df_enhanced.columns if 'rolling' in col]
    enhanced_features.extend(rolling_features)
    
    # Add z-score features
    zscore_features = [col for col in df_enhanced.columns if 'zscore' in col]
    enhanced_features.extend(zscore_features)
    
    # Remove any missing features
    enhanced_features = [f for f in enhanced_features if f in df_enhanced.columns]
    
    print(f"\n🔧 Selected features: {enhanced_features}")
    
    # Preprocess features
    X_scaled, scaler, valid_indices = preprocess_features(df_enhanced, enhanced_features)
    df_valid = df_enhanced.loc[valid_indices].reset_index(drop=True)
    
    # Train ensemble model
    results = ensemble_anomaly_detection(X_scaled, contamination=0.05)
    
    # Evaluate each model
    for model_name, predictions in results.items():
        if model_name not in ['iso_scores', 'lof_scores', 'models']:
            evaluate_model(predictions, predictions, model_name.replace('_', ' ').title())
    
    # Use ensemble predictions for final results
    final_predictions = results['ensemble']
    
    # Add results to dataframe
    df_valid['anomaly'] = final_predictions
    df_valid['anomaly_label'] = df_valid['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    df_valid['iso_score'] = results['iso_scores']
    df_valid['lof_score'] = results['lof_scores']
    
    # Save detailed results
    output_columns = ['pc_speed', 'pc_steering', 'pc_brake', 'anomaly_label', 'iso_score', 'lof_score']
    df_valid[output_columns].to_csv("improved_annotated_output.csv", index=False)
    
    # Create visualizations
    visualize_anomalies(df_valid, final_predictions, results['iso_scores'])
    
    # Summary statistics
    print("\n📊 Final Results Summary:")
    print(f"Total samples processed: {len(df_valid)}")
    print(f"Anomalies detected: {np.sum(final_predictions == -1)} ({np.sum(final_predictions == -1)/len(final_predictions)*100:.2f}%)")
    print(f"Normal samples: {np.sum(final_predictions == 1)} ({np.sum(final_predictions == 1)/len(final_predictions)*100:.2f}%)")
    
    # Show most anomalous samples
    print("\n🚨 Top 5 Most Anomalous Samples:")
    anomaly_indices = df_valid[df_valid['anomaly'] == -1].index
    if len(anomaly_indices) > 0:
        top_anomalies = df_valid.loc[anomaly_indices].nsmallest(5, 'iso_score')
        print(top_anomalies[['pc_speed', 'pc_steering', 'pc_brake', 'iso_score']].to_string())
    
    print("\n✅ Enhanced anomaly detection complete!")
    print("📁 Files saved:")
    print("  - improved_annotated_output.csv (detailed results)")
    print("  - improved_anomaly_plot.png (visualizations)")

if __name__ == "__main__":
    main()
