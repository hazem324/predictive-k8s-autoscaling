import pickle
import numpy as np

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Fake feature vector (24 features)
sample = np.array([[
    10,20,30,40,5,
    100,0.01,0.005,
    14,2,0,
    9,8,7,5,4,
    12,15,1.2,6,
    3,2,0,1
]])

prediction = model.predict(sample)

print("Prediction:", prediction)