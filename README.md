# AI-Powered Anomaly Detection for Software-Defined Vehicles (SDVs)

This project uses unsupervised Machine Learning (Isolation Forest) to detect abnormal driving patterns from real-time or recorded telemetry logs. It’s designed as part of the Tata Technologies InnoVent Hackathon under the track **“AI-powered virtual testing environments for SDVs”**.


### OUTPUT
scatter plot with red anomalies and green normal points

![](https://github.com/Kash1444/sdv-anomaly-detection/blob/5ea8d94c5d100cb51cbee9352ad2d338fe83cd72/output1.png)
---

## Features

- Fast anomaly detection using Isolation Forest
- Analyzes speed, steering angle, and brake input
- Flags unsafe or unusual driving behavior
- Outputs labeled CSV for further testing or dashboard integration

---

## Tech Stack

- Python 3.11
- pandas
- scikit-learn
- (Optional) Streamlit or Flask for UI

---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/Kash1444/sdv-anomaly-detection.git
cd sdv-anomaly-detection
