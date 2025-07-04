import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the annotated data
df = pd.read_csv("annotated_output.csv")

# Set style
sns.set(style="whitegrid")

# Scatter plot: Speed vs Steering with anomaly color
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="pc_speed",
    y="pc_steering",
    hue="anomaly",
    palette={"Normal": "green", "Anomaly": "red"},
    alpha=0.7
)

plt.title("Driving Data Anomaly Detection")
plt.xlabel("Speed (km/h)")
plt.ylabel("Steering Angle (degrees)")
plt.legend(title="Status")
plt.tight_layout()
plt.savefig("anomaly_plot.png")
plt.show()
