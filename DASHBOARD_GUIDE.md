# 🚀 Quick Start Guide for SDV Anomaly Detection Dashboard

## Run the Interactive Dashboard

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
```bash
streamlit run streamlit_dashboard.py
```

### 3. Open in Browser
The dashboard will automatically open at: `http://localhost:8501`

## Dashboard Features

### 📊 **Interactive Analysis**
- Upload your own CSV files or use sample data
- Adjust detection sensitivity with contamination slider
- Toggle enhanced feature engineering
- Real-time anomaly detection

### 📈 **Rich Visualizations**
- **Scatter Plots**: Speed vs Steering/Brake with anomaly highlighting
- **Time Series**: Driving patterns over time with anomaly markers
- **Score Distributions**: Confidence scores for detected anomalies
- **Top Anomalies**: Most suspicious driving behaviors

### 💾 **Export Results**
- Download results as CSV or JSON
- Timestamped filenames for easy tracking
- Ready for further analysis or reporting

## Usage Tips

1. **Upload Format**: Ensure your CSV has columns: `pc_speed`, `pc_steering`, `pc_brake`
2. **Contamination**: Start with 5% (0.05) and adjust based on results
3. **Enhanced Features**: Keep enabled for better detection accuracy
4. **File Size**: Works best with files under 10MB for smooth performance

## Next Steps

Once you have the dashboard running, try:
- Testing with different contamination rates
- Uploading your own driving data
- Comparing basic vs enhanced feature detection
- Downloading results for further analysis

## Troubleshooting

**Issue**: Dashboard won't start
- Solution: Ensure all requirements are installed: `pip install -r requirements.txt`

**Issue**: "File not found" error
- Solution: Either upload a CSV file or ensure `realistic_driving_data.csv` exists

**Issue**: Slow performance
- Solution: Try smaller datasets or reduce contamination rate for faster processing

---

**🎯 Ready to detect anomalies? Run `streamlit run streamlit_dashboard.py` and start analyzing!**
