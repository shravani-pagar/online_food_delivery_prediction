import joblib

model = joblib.load("best_model.pkl")

print(type(model))
print("Model loaded successfully!")