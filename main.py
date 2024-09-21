import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load the saved model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the original dataset for graph plotting (adjust path to your dataset)
df = pd.read_csv(r'D:/train.csv')

# Prepare features for training and testing to assess performance
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']].fillna(0)
y_actual = df['SalePrice']

# Predict using the model
y_pred_all = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y_actual, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred_all)

# Streamlit app for predicting house prices
st.title("House Price Prediction")

# Create input fields for square footage, bedrooms, and bathrooms
square_footage = st.number_input("Enter Square Footage", min_value=500, max_value=10000, value=2000, step=100)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Enter Number of Full Bathrooms", min_value=1, max_value=5, value=2, step=1)

# Create a button for predicting the house price
if st.button("Predict Price"):
    # Prepare the input for the model
    new_house = pd.DataFrame({
        'GrLivArea': [square_footage],
        'BedroomAbvGr': [bedrooms],
        'FullBath': [bathrooms]
    })

    # Predict the house price using the loaded model
    predicted_price = model.predict(new_house)

    # Display the predicted price
    st.write(f"The predicted price for the house is: ${predicted_price[0]:,.2f}")

    # --- Plotting ---
    # Scatter plot for GrLivArea vs SalePrice
    st.write("### Relationship between Square Footage and House Prices")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', ax=ax, alpha=0.5)
    ax.set_title("Square Footage vs House Prices")
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Sale Price")
    st.pyplot(fig)

    # Display a histogram of the house prices
    st.write("### Distribution of House Prices")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['SalePrice'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribution of Sale Prices")
    st.pyplot(fig2)

    # --- Additional Visualization ---
    # Scatter plot of actual vs predicted prices
    st.write("### Predicted vs Actual Prices")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_actual, y=y_pred_all, ax=ax3, alpha=0.6)
    ax3.set_title("Actual vs Predicted House Prices")
    ax3.set_xlabel("Actual Sale Price")
    ax3.set_ylabel("Predicted Sale Price")
    st.pyplot(fig3)

# Display the model performance metrics
st.write("## Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.4f}")
