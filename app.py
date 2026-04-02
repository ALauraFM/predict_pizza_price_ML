import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("pizzas.csv")

model = LinearRegression()
x = df[["diameter"]]
y = df["price"]

model.fit(x, y)
st.title("Pizza Price Predictor")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(
        "https://img.freepik.com/vetores-gratis/pizza-slice-felted-floating-cartoon-vector-icon-ilustracao-icon-objeto-alimentar-vector-plano-isolado_138676-10422.jpg",
        width=200
    )

st.markdown("""
This app uses a Linear Regression model to predict the price of a pizza based on its diameter.  
Enter a value below and get an instant prediction.
""")

st.divider()

diameter = st.number_input("Enter the diameter of the pizza (cm)", min_value=0, step=1, key="diameter_input")

   
if st.button("Predict Price"):
    predicted_price = model.predict([[diameter]])[0]
    st.write(f"The predicted price for a pizza with a diameter of {diameter} cm is: R${predicted_price:.2f}")
    
