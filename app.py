import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==2.1.4", "numpy==1.26.4", "joblib==1.4.2"])

import streamlit as st
import pandas as pd
import numpy as np
import joblib
# ==========================================
# Shopping Cart
# ==========================================

if "cart" not in st.session_state:
    st.session_state.cart = []

st.set_page_config(
    page_title="Online Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)

# Load dataset
df = pd.read_csv("dataset.csv")   # or your actual CSV filename

# Load model safely
try:
    model = joblib.load("best_model.pkl")
    restaurant_encoder = joblib.load("restaurant_encoder.pkl")
    payment_encoder = joblib.load("payment_encoder.pkl")
    status_encoder = joblib.load("status_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()
# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#FFF8F0;
}

h1,h2,h3{
    color:#FC8019;
}

.stButton>button{
    background:#FC8019;
    color:white;
    border-radius:10px;
    border:none;
    font-size:18px;
}

.stButton>button:hover{
    background:#ff5c00;
}

.sidebar .sidebar-content{
    background:#FFF3E0;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🍔 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualization",
        "🍽 Menu",
        "🛒 Cart",
        "🤖 Prediction",
        "ℹ About"
    ]
)
# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🍔 Online Food Delivery Time Prediction")

    st.image(
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
    )

    st.markdown("---")

    st.header("📌 About Project")

    st.write("""
This project predicts the estimated delivery time of an online food order
using Machine Learning.

The prediction is based on customer details, restaurant type,
delivery distance, payment method, delivery partner rating,
and order information.
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric("Target", "Delivery Time")

    st.markdown("---")

    st.success("Use the sidebar to explore the dataset and make predictions.")

    # ==========================================
# DATASET PAGE
# ==========================================

elif page == "📊 Dataset":

    st.title("📊 Dataset")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shape")
        st.write(df.shape)

    with col2:
        st.subheader("Columns")
        st.write(df.columns.tolist())

    st.markdown("---")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.markdown("---")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # ==========================================
# VISUALIZATION PAGE
# ==========================================

elif page == "📈 Visualization":

    st.title("📈 Data Visualization")

    chart = st.selectbox(
        "Select Chart",
        (
            "Customer Age Distribution",
            "Restaurant Type",
            "Payment Method",
            "Order Status",
            "Delivery Distance",
            "Delivery Time"
        )
    )

    if chart == "Customer Age Distribution":

        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df["customer_age"], bins=20)
        ax.set_xlabel("Customer Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif chart == "Restaurant Type":

        fig, ax = plt.subplots(figsize=(8,5))
        df["restaurant_type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    elif chart == "Payment Method":

        fig, ax = plt.subplots(figsize=(8,5))
        df["payment_method"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    elif chart == "Order Status":

        fig, ax = plt.subplots(figsize=(8,5))
        df["order_status"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

    elif chart == "Delivery Distance":

        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df["delivery_distance_km"], df["delivery_time_minutes"])
        ax.set_xlabel("Distance")
        ax.set_ylabel("Delivery Time")
        st.pyplot(fig)

    elif chart == "Delivery Time":

        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df["delivery_time_minutes"], bins=20)
        st.pyplot(fig)

# ==========================================
# ==========================================
# MENU PAGE
# ==========================================

elif page == "🍽 Menu":

    st.markdown("""
    <style>
    .menu-title{
        background:linear-gradient(90deg,#FC8019,#FF9800);
        padding:15px;
        border-radius:12px;
        color:white;
        text-align:center;
        font-size:35px;
        font-weight:bold;
    }

    .food-card{
        background:white;
        border:2px solid #FC8019;
        border-radius:15px;
        padding:10px;
        text-align:center;
        box-shadow:2px 2px 8px #d3d3d3;
        margin-bottom:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='menu-title'>🍽 Our Delicious Menu</div>",
        unsafe_allow_html=True
    )

    st.write("### Fresh • Tasty • Fast Delivery 🚀")
    st.markdown("---")

    foods = [
        ("🍔 Burger","₹199","⭐ 4.8","https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500"),
        ("🍕 Pizza","₹349","⭐ 4.9","https://images.unsplash.com/photo-1513104890138-7c749659a591?w=500"),
        ("🍝 Pasta","₹249","⭐ 4.7","https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=500"),
        ("🍟 French Fries","₹129","⭐ 4.6","https://images.unsplash.com/photo-1576107232684-1279f390859f?w=500"),
        ("🌮 Tacos","₹229","⭐ 4.8","https://images.unsplash.com/photo-1552332386-f8dd00dc2f85?w=500"),
        ("🥗 Salad","₹149","⭐ 4.5","https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=500"),
        ("🍰 Cake","₹299","⭐ 4.9","https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=500"),
        ("🥤 Cold Drink","₹79","⭐ 4.4","https://images.unsplash.com/photo-1544145945-f90425340c7e?w=500"),
        ("🍦 Ice Cream","₹99","⭐ 4.8","https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=500")
    ]

    for i in range(0, len(foods), 3):

     col1, col2, col3 = st.columns(3)

    for col, food in zip([col1, col2, col3], foods[i:i+3]):

        name, price, rating, image = food

        with col:

            st.image(image)

            st.markdown(f"""
            <div class='food-card'>
                <h3>{name}</h3>
                <h4 style='color:green'>{price}</h4>
                <p>{rating}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🛒 Order Now", key=name):

                st.session_state.cart.append({
                    "Food": name,
                    "Price": int(price.replace("₹", ""))
                })

                st.success(f"✅ {name} added to cart!")

                st.balloons()

                st.info(f"""
🍽 Food : {name}

💰 Price : {price}

⭐ Rating : {rating}

🚴 Estimated Delivery : 30–40 Minutes
""")
                
# ==========================================
# CART PAGE
# ==========================================

elif page == "🛒 Cart":

    st.title("🛒 Shopping Cart")

    if len(st.session_state.cart) == 0:
        st.warning("Your cart is empty.")

    else:

        total = 0

        for i, item in enumerate(st.session_state.cart):

            col1, col2, col3 = st.columns([4,2,1])

            with col1:
                st.write(item["Food"])

            with col2:
                st.write(f"₹{item['Price']}")

            with col3:
                if st.button("❌", key=f"remove{i}"):
                    st.session_state.cart.pop(i)
                    st.rerun()

            total += item["Price"]

        st.markdown("---")

        st.subheader(f"💰 Total Amount : ₹{total}")

        if st.button("✅ Place Order"):

            st.balloons()
            st.success("🎉 Order Placed Successfully!")
            st.session_state.cart = []
# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "🤖 Prediction":

    st.markdown("""
    <style>
    .prediction-box{
        background: linear-gradient(90deg,#FC8019,#FFB347);
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        font-size:30px;
        font-weight:bold;
        margin-bottom:20px;
    }

    .result-box{
        background:#E8F5E9;
        border-left:8px solid green;
        padding:20px;
        border-radius:10px;
        font-size:24px;
        color:#1B5E20;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='prediction-box'>🍔 Food Delivery Time Prediction 🚴</div>",
        unsafe_allow_html=True
    )

    st.write("### Enter Order Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "👤 Customer Age",
            min_value=18,
            max_value=80,
            value=25
        )

        restaurant = st.selectbox(
            "🍽 Restaurant Type",
            restaurant_encoder.classes_
        )

        order_value = st.number_input(
            "💰 Order Value (₹)",
            min_value=50.0,
            value=300.0
        )

        distance = st.number_input(
            "📍 Delivery Distance (KM)",
            min_value=0.5,
            value=5.0
        )

        payment = st.selectbox(
            "💳 Payment Method",
            payment_encoder.classes_
        )

    with col2:

        rating = st.slider(
            "⭐ Delivery Partner Rating",
            1.0,
            5.0,
            4.0
        )

        status = st.selectbox(
            "📦 Order Status",
            status_encoder.classes_
        )

        year = st.number_input(
            "📅 Year",
            value=2024
        )

        month = st.slider(
            "📆 Month",
            1,
            12,
            6
        )

        day = st.slider(
            "📍 Day",
            1,
            31,
            15
        )

        weekday = st.selectbox(
            "📅 Day of Week",
            [0,1,2,3,4,5,6],
            format_func=lambda x:
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict Delivery Time"):

        restaurant = restaurant_encoder.transform([restaurant])[0]
        payment = payment_encoder.transform([payment])[0]
        status = status_encoder.transform([status])[0]

        prediction = model.predict([[
            age,
            restaurant,
            order_value,
            distance,
            payment,
            rating,
            status,
            year,
            month,
            day,
            weekday
        ]])

        st.balloons()

        st.markdown(
            f"""
            <div class='result-box'>
            ⏰ Estimated Delivery Time<br><br>
            {prediction[0]:.2f} Minutes
            </div>
            """,
            unsafe_allow_html=True
        )

        
