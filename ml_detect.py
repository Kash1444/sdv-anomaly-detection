from sklearn.ensemble import IsolationForest
import pandas as pd

df = pd.read_csv("realistic_driving_data.csv")

features = ['pc_speed', 'pc_steering', 'pc_brake']
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df[features])
df['anomaly'] = model.predict(df[features])
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

df.to_csv("annotated_output.csv", index=False)
print(" Anomaly detection complete! Output saved to annotated_output.csv")
