# ==========================================
# Online Food Delivery Time Prediction
# Model Training Script
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# ==========================================
# Check Missing Values
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# Remove missing values if any
df.dropna(inplace=True)

# ==========================================
# Remove Duplicate Rows
# ==========================================

df.drop_duplicates(inplace=True)

# ==========================================
# Convert Date Column
# ==========================================

df["order_date"] = pd.to_datetime(df["order_date"])

# ==========================================
# Feature Engineering
# ==========================================

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_day"] = df["order_date"].dt.day
df["day_of_week"] = df["order_date"].dt.dayofweek

# Drop original date
df.drop("order_date", axis=1, inplace=True)

# ==========================================
# Drop ID Column
# ==========================================

df.drop("order_id", axis=1, inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================

restaurant_encoder = LabelEncoder()
payment_encoder = LabelEncoder()
status_encoder = LabelEncoder()

df["restaurant_type"] = restaurant_encoder.fit_transform(df["restaurant_type"])
df["payment_method"] = payment_encoder.fit_transform(df["payment_method"])
df["order_status"] = status_encoder.fit_transform(df["order_status"])

# ==========================================
# Save Encoders
# ==========================================

joblib.dump(restaurant_encoder, "restaurant_encoder.pkl")
joblib.dump(payment_encoder, "payment_encoder.pkl")
joblib.dump(status_encoder, "status_encoder.pkl")

print("\nEncoders Saved Successfully!")

# ==========================================
# Features and Target
# ==========================================

X = df.drop("delivery_time_minutes", axis=1)
y = df["delivery_time_minutes"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================
# Models
# ==========================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

best_model = None
best_score = -999
best_name = ""

print("\n==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, prediction)

    print("\n", "=" * 45)
    print(name)
    print("=" * 45)

    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("\n==============================")
print("Best Model :", best_name)
print("Best R2 Score :", round(best_score, 4))
print("==============================")

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(best_model, "best_model.pkl")

print("\nModel Saved Successfully!")
print("File Name : best_model.pkl")

# ==========================================
# Feature Names
# ==========================================

print("\nFeatures Used")

for column in X.columns:
    print(column)

print("\nTraining Completed Successfully!")