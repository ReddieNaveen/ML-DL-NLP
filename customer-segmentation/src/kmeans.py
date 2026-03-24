import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/Mall_Customers.csv")

# Drop non-useful columns
df = df.drop("CustomerID", axis=1)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot elbow
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Final model (choose 5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Visualization
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c=df["Cluster"])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

### Key Insights

# - Customers can be grouped based on income and spending behavior  
# - High income + high spending → premium customers  
# - High income + low spending → target customers for marketing  
# - Low income + high spending → impulsive buyers  
# - Low income + low spending → low-value customers  