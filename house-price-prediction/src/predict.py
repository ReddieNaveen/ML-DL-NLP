import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)


sample = np.array([[60000, 5, 7, 3, 30000]])

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print(f"Predicted House Price: {prediction[0]}")