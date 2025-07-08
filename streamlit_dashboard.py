"""
🚗 SDV Anomaly Detection Dashboard
Interactive Streamlit web application for real-time driving anomaly detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import warnings
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SDV Anomaly Detection Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .alert-danger {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border-left: 4px solid #c62828;
    }
    .alert-success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_samples=1000):
    """Generate synthetic driving data"""
    np.random.seed(42)
    
    # Normal driving patterns
    normal_samples = int(n_samples * 0.95)
    
    # Speed: mostly 30-80 km/h
    speed_normal = np.random.normal(50, 15, normal_samples)
    speed_normal = np.clip(speed_normal, 0, 120)
    
    # Steering: mostly small values
    steering_normal = np.random.normal(0, 0.1, normal_samples)
    steering_normal = np.clip(steering_normal, -1, 1)
    
    # Brake: mostly low values
    brake_normal = np.random.exponential(0.05, normal_samples)
    brake_normal = np.clip(brake_normal, 0, 1)
    
    # Anomalous patterns
    anomaly_samples = n_samples - normal_samples
    
    # Speed anomalies: very high or very low
    speed_anomaly = np.concatenate([
        np.random.normal(120, 10, anomaly_samples//2),
        np.random.normal(10, 5, anomaly_samples//2)
    ])
    speed_anomaly = np.clip(speed_anomaly, 0, 150)
    
    # Steering anomalies: extreme values
    steering_anomaly = np.random.choice([-1, 1], anomaly_samples) * np.random.uniform(0.7, 1.0, anomaly_samples)
    
    # Brake anomalies: hard braking
    brake_anomaly = np.random.uniform(0.7, 1.0, anomaly_samples)
    
    # Combine data
    speed = np.concatenate([speed_normal, speed_anomaly])
    steering = np.concatenate([steering_normal, steering_anomaly])
    brake = np.concatenate([brake_normal, brake_anomaly])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    
    df = pd.DataFrame({
        'pc_speed': speed[indices],
        'pc_steering': steering[indices],
        'pc_brake': brake[indices]
    })
    
    return df

def feature_engineering(df):
    """Create enhanced features"""
    df_processed = df.copy()
    
    # Speed categories
    df_processed['speed_category'] = pd.cut(df_processed['pc_speed'], 
                                          bins=[0, 30, 60, 90, 200], 
                                          labels=['Low', 'Medium', 'High', 'Very High'])
    df_processed['speed_category_encoded'] = df_processed['speed_category'].cat.codes
    
    # Aggressive behaviors
    df_processed['aggressive_steering'] = (abs(df_processed['pc_steering']) > 0.5).astype(int)
    df_processed['hard_braking'] = (df_processed['pc_brake'] > 0.7).astype(int)
    
    # Ratios
    df_processed['speed_steering_ratio'] = df_processed['pc_speed'] / (abs(df_processed['pc_steering']) + 0.001)
    df_processed['brake_intensity'] = df_processed['pc_brake'] * df_processed['pc_speed']
    
    # Rolling features
    df_processed['speed_rolling_mean'] = df_processed['pc_speed'].rolling(window=5, min_periods=1).mean()
    df_processed['speed_rolling_std'] = df_processed['pc_speed'].rolling(window=5, min_periods=1).std().fillna(0)
    
    # Z-scores
    df_processed['speed_zscore'] = (df_processed['pc_speed'] - df_processed['pc_speed'].mean()) / df_processed['pc_speed'].std()
    df_processed['steering_zscore'] = (df_processed['pc_steering'] - df_processed['pc_steering'].mean()) / df_processed['pc_steering'].std()
    
    return df_processed

def preprocess_features(df, feature_cols):
    """Preprocess features for ML"""
    # Remove any missing values
    df_clean = df[feature_cols].dropna()
    valid_indices = df_clean.index
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    return X_scaled, scaler, valid_indices

def ensemble_anomaly_detection(X_scaled, contamination=0.05):
    """Run ensemble anomaly detection"""
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso_forest.fit_predict(X_scaled)
    iso_scores = iso_forest.score_samples(X_scaled)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination, novelty=False)
    lof_labels = lof.fit_predict(X_scaled)
    lof_scores = lof.negative_outlier_factor_
    
    # Ensemble: both methods agree
    ensemble_labels = np.where((iso_labels == -1) & (lof_labels == -1), -1, 1)
    
    return {
        'ensemble': ensemble_labels,
        'iso_scores': iso_scores,
        'lof_scores': lof_scores,
        'iso_labels': iso_labels,
        'lof_labels': lof_labels
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 SDV Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # File upload or use sample data
    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["Sample Data", "Upload CSV"]
    )
    
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload driving data CSV",
            type=['csv'],
            help="CSV should contain columns: pc_speed, pc_steering, pc_brake"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ File uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                return
    else:
        # Generate sample data
        df = generate_sample_data(1000)
        st.sidebar.info(f"📊 Sample data generated: {df.shape[0]} rows")
    
    if df is None:
        st.info("👆 Please select a data source from the sidebar to begin analysis.")
        return
    
    # Validate required columns
    required_cols = ['pc_speed', 'pc_steering', 'pc_brake']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ CSV must contain columns: {required_cols}")
        st.info(f"Found columns: {list(df.columns)}")
        return
    
    # Analysis parameters
    st.sidebar.subheader("🔧 Detection Parameters")
    contamination = st.sidebar.slider(
        "Contamination Rate", 
        min_value=0.01, 
        max_value=0.20, 
        value=0.05, 
        step=0.01,
        help="Expected percentage of anomalies in the data"
    )
    
    use_enhanced_features = st.sidebar.checkbox(
        "Use Enhanced Features", 
        value=True,
        help="Include engineered features for better detection"
    )
    
    # Run Analysis Button
    if st.sidebar.button("🚀 Run Anomaly Detection", type="primary"):
        with st.spinner("Analyzing driving data..."):
            results = run_anomaly_detection(df, contamination, use_enhanced_features)
            
            if results:
                display_results(results)
    
    # Data preview
    st.subheader("📊 Data Preview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Date Range", f"{len(df)} time points")
    
    # Show data
    st.dataframe(df.head(100), use_container_width=True)
    
    # Basic statistics
    if st.expander("📈 Data Statistics"):
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(required_cols):
            axes[i].hist(df[col], bins=30, alpha=0.7, color=['blue', 'green', 'red'][i])
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)

def run_anomaly_detection(df, contamination, use_enhanced_features):
    """Run the anomaly detection pipeline"""
    try:
        # Feature engineering
        if use_enhanced_features:
            df_processed = feature_engineering(df)
            feature_cols = ['pc_speed', 'pc_steering', 'pc_brake', 'speed_category_encoded',
                           'aggressive_steering', 'hard_braking', 'speed_steering_ratio', 
                           'brake_intensity', 'speed_rolling_mean', 'speed_rolling_std',
                           'speed_zscore', 'steering_zscore']
        else:
            df_processed = df.copy()
            feature_cols = ['pc_speed', 'pc_steering', 'pc_brake']
        
        # Preprocess features
        X_scaled, scaler, valid_indices = preprocess_features(df_processed, feature_cols)
        df_valid = df_processed.loc[valid_indices].reset_index(drop=True)
        
        # Run ensemble detection
        detection_results = ensemble_anomaly_detection(X_scaled, contamination=contamination)
        
        # Add results to dataframe
        df_valid['anomaly'] = detection_results['ensemble']
        df_valid['anomaly_label'] = df_valid['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        df_valid['iso_score'] = detection_results['iso_scores']
        df_valid['lof_score'] = detection_results['lof_scores']
        
        return {
            'df_results': df_valid,
            'detection_results': detection_results,
            'feature_cols': feature_cols,
            'contamination': contamination
        }
        
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        return None

def display_results(results):
    """Display analysis results"""
    df_results = results['df_results']
    
    st.success("✅ Anomaly detection completed successfully!")
    
    # Key metrics
    st.subheader("🎯 Detection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = len(df_results)
    anomalies = np.sum(df_results['anomaly'] == -1)
    normal_samples = np.sum(df_results['anomaly'] == 1)
    anomaly_rate = (anomalies / total_samples) * 100
    
    with col1:
        st.metric("Total Samples", total_samples)
    with col2:
        st.metric("Anomalies Detected", anomalies, delta=f"{anomaly_rate:.1f}%")
    with col3:
        st.metric("Normal Samples", normal_samples)
    with col4:
        st.metric("Detection Rate", f"{anomaly_rate:.1f}%")
    
    # Alert if high anomaly rate
    if anomaly_rate > 15:
        st.markdown(
            '<div class="alert-danger">⚠️ High anomaly rate detected! Consider adjusting contamination parameter.</div>',
            unsafe_allow_html=True
        )
    elif anomaly_rate < 1:
        st.markdown(
            '<div class="alert-danger">⚠️ Very low anomaly rate. Consider lowering contamination parameter.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="alert-success">✅ Anomaly detection rate looks reasonable.</div>',
            unsafe_allow_html=True
        )
    
    # Interactive visualizations
    st.subheader("📊 Interactive Visualizations")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plots", "Time Series", "Score Distribution", "Top Anomalies"])
    
    with tab1:
        create_scatter_plots(df_results)
    
    with tab2:
        create_time_series_plot(df_results)
    
    with tab3:
        create_score_distribution(df_results)
    
    with tab4:
        show_top_anomalies(df_results)

def create_scatter_plots(df_results):
    """Create interactive scatter plots"""
    
    # Speed vs Steering
    fig1 = px.scatter(
        df_results, 
        x='pc_speed', 
        y='pc_steering',
        color='anomaly_label',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title="Speed vs Steering Analysis",
        hover_data=['iso_score', 'lof_score']
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Speed vs Brake
    fig2 = px.scatter(
        df_results, 
        x='pc_speed', 
        y='pc_brake',
        color='anomaly_label',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title="Speed vs Brake Analysis",
        hover_data=['iso_score', 'lof_score']
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

def create_time_series_plot(df_results):
    """Create time series visualization"""
    
    # Add index as time proxy
    df_results['time_index'] = range(len(df_results))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Speed Over Time', 'Steering Over Time', 'Brake Over Time'),
        vertical_spacing=0.05
    )
    
    # Speed
    fig.add_trace(
        go.Scatter(x=df_results['time_index'], y=df_results['pc_speed'], 
                  mode='lines', name='Speed', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Highlight anomalies
    anomaly_indices = df_results[df_results['anomaly'] == -1]['time_index']
    if len(anomaly_indices) > 0:
        anomaly_speeds = df_results[df_results['anomaly'] == -1]['pc_speed']
        
        fig.add_trace(
            go.Scatter(x=anomaly_indices, y=anomaly_speeds,
                      mode='markers', name='Speed Anomalies', 
                      marker=dict(color='red', size=8)),
            row=1, col=1
        )
    
    # Steering
    fig.add_trace(
        go.Scatter(x=df_results['time_index'], y=df_results['pc_steering'],
                  mode='lines', name='Steering', line=dict(color='green')),
        row=2, col=1
    )
    
    if len(anomaly_indices) > 0:
        anomaly_steering = df_results[df_results['anomaly'] == -1]['pc_steering']
        fig.add_trace(
            go.Scatter(x=anomaly_indices, y=anomaly_steering,
                      mode='markers', name='Steering Anomalies',
                      marker=dict(color='red', size=8)),
            row=2, col=1
        )
    
    # Brake
    fig.add_trace(
        go.Scatter(x=df_results['time_index'], y=df_results['pc_brake'],
                  mode='lines', name='Brake', line=dict(color='orange')),
        row=3, col=1
    )
    
    if len(anomaly_indices) > 0:
        anomaly_brake = df_results[df_results['anomaly'] == -1]['pc_brake']
        fig.add_trace(
            go.Scatter(x=anomaly_indices, y=anomaly_brake,
                      mode='markers', name='Brake Anomalies',
                      marker=dict(color='red', size=8)),
            row=3, col=1
        )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time Index", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def create_score_distribution(df_results):
    """Create anomaly score distribution plots"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            df_results, 
            x='iso_score',
            color='anomaly_label',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            title="Isolation Forest Score Distribution",
            nbins=30
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(
            df_results, 
            x='lof_score',
            color='anomaly_label',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            title="LOF Score Distribution",
            nbins=30
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

def show_top_anomalies(df_results):
    """Show top anomalous samples"""
    
    st.subheader("🚨 Most Anomalous Samples")
    
    anomaly_data = df_results[df_results['anomaly'] == -1]
    
    if len(anomaly_data) > 0:
        # Sort by isolation forest score (most negative = most anomalous)
        top_anomalies = anomaly_data.nsmallest(10, 'iso_score')
        
        # Display in a nice table
        display_cols = ['pc_speed', 'pc_steering', 'pc_brake', 'iso_score', 'lof_score']
        st.dataframe(
            top_anomalies[display_cols].round(4),
            use_container_width=True
        )
        
        # Show statistics
        st.subheader("📊 Anomaly Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Avg Speed (Anomalies)", 
                f"{anomaly_data['pc_speed'].mean():.2f}",
                delta=f"{anomaly_data['pc_speed'].mean() - df_results['pc_speed'].mean():.2f}"
            )
        
        with col2:
            st.metric(
                "Avg |Steering| (Anomalies)", 
                f"{abs(anomaly_data['pc_steering']).mean():.2f}",
                delta=f"{abs(anomaly_data['pc_steering']).mean() - abs(df_results['pc_steering']).mean():.2f}"
            )
        
        with col3:
            st.metric(
                "Avg Brake (Anomalies)", 
                f"{anomaly_data['pc_brake'].mean():.2f}",
                delta=f"{anomaly_data['pc_brake'].mean() - df_results['pc_brake'].mean():.2f}"
            )
    else:
        st.info("No anomalies detected with current parameters.")

# FIXED: Proper main function call
if __name__ == "__main__":
    main()