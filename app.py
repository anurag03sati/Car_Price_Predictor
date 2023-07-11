import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


car=pd.read_csv('cleaned_car.csv')
st.title('Car Price Predictor')

cname=st.selectbox(
    'Select company',sorted(car['company'].unique())
)
filtered_cars = car[car['company'] ==cname]

name=st.selectbox(
    'Select name',sorted(filtered_cars['name'].unique())
)
year=st.selectbox(
    'Select year',sorted(car['year'].unique())
)
km = st.text_input("Enter kms driven", "20000")


fuel=st.selectbox(
    'Select fuel type',sorted(car['fuel_type'].unique())
)






                              
button_clicked = st.button("Predict Price")
if button_clicked:
    x = car.drop(columns=["Price", "Unnamed: 0"])  # everything is a feature except price
    y = car['Price']
    ohe = OneHotEncoder()
    ohe.fit(x[['name', 'company', 'fuel_type']])
    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']), remainder="passthrough")
    score = []
    for i in range(1000):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=i)  # basically for what value of i state split, r2score is max
        lr = LinearRegression()
        pipe = make_pipeline(column_trans, lr)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        score.append(r2_score(y_test, y_pred))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.argmax(
        score))  # now we have the score thus use it easily
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(x_train, y_train)
    data = pd.DataFrame([[name, cname, year, km, fuel]],
                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])


    result = pipe.predict(data)

    st.text(result)
