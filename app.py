import streamlit as st
import numpy as np
import pickle

# Import Model
#rf = pickle.load(open("D:/PIAIC COURSE/Miss Aqsa/app/yieldApp/w_rfm_model.pkl", 'rb'))
xg = pickle.load(open("D:/PIAIC COURSE/Miss Aqsa/app/yieldApp/w_xgb_model.pkl", 'rb'))
lgbm = pickle.load(open("D:/PIAIC COURSE/Miss Aqsa/app/yieldApp/w_gbm_model.pkl", 'rb'))


st.title("Yield Predictor")
# Select Value with Slider

clone_size = st.slider("Select Clone Size", 10, 40, 20)
bumbles = st.slider("Select bumbles", 0.0, 0.585, 0.10)
andrena = st.slider("Select andrena", 0.0, 0.750, 0.23)
osmia = st.slider("Select osmia", 0.0, 0.750, 0.15)
AverageOfUpperTRange = st.slider("Select AverageOfUpperTRange", 58.20, 79.0, 20.0)
AverageOfLowerTRange = st.slider("Select AverageOfLowerTRange", 41.20, 55.90, 20.0)
AverageRainingDays = st.slider("Select AverageRainingDays", 0.060, 0.560, 0.24)

# Select Value with manual

# clone_size = st.number_input("Enter Clone Size", max_value=40.0 , key="clone_size")
# bumbles = st.number_input("Enter Bumbles:" ,key="bumbles")
# andrena = st.number_input("Enter Andrena:" , key="andrena")
# osmia = st.number_input("Enter Osmia:" , key="osmia")
# AverageOfUpperTRange = st.number_input("Enter AverageOfUpperTRange:" , key="AverageOfUpperTRange")
# AverageOfLowerTRange = st.number_input("Enter AverageOfLowerTRange:" , key="AverageOfLowerTRange")
# AverageRainingDays = st.number_input("Enter AverageRainingDays:" , key="AverageRainingDays")

Submit = st.button('Predict Yield')
if Submit:
    query = np.array([clone_size , bumbles ,andrena ,
                      osmia , AverageOfUpperTRange ,
                      AverageOfLowerTRange , AverageRainingDays])
    query = query.reshape(1,7)

    #st.title("The Predicted Yield From RandomForest is " + str(int(rf.predict(query)[0])))

    st.title("The Predicted Yield From XGBoost is " + str(int(xg.predict(query)[0])))

    st.title("The Predicted Yield From LightGBM is " + str(int(lgbm.predict(query)[0])))
