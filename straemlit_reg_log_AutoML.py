#app for the 2 options : input parameters or load data file

import streamlit as st

import numpy as np
import pandas as pd

import pickle
import sklearn



st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
## MSDE4 : AutoML Model: Logistic Regression to Predict Diabeties
###This app Define the **Clients** Cluster 
""")

st.sidebar.image("https://img.passeportsante.net/1200x675/2021-07-23/shutterstock-1439349791.webp",width=300)

option = st.selectbox(
     'How would you like to use the prediction model?',
     ('','input parameters directly', 'Load a file of data'))


def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', min_value=0,max_value=374)
    glucose = st.sidebar.slider('Glucose', min_value=0,max_value=7847)
    bloodPressure = st.sidebar.slider('BloodPressure in €', min_value=0,max_value=280206)
    skinThickness = st.sidebar.slider('SkinThickness in €', min_value=0,max_value=280206)
    insulin = st.sidebar.slider('Insulin in €', min_value=0,max_value=280206)
    bmi = st.sidebar.slider('BMI in €', min_value=0,max_value=280206)
    diabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction in €', min_value=0,max_value=280206)
    age = st.sidebar.slider('Age in €', min_value=0,max_value=280206)
    
    data = {
    	    'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bloodPressure,
            'SkinThickness': skinThickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetesPedigreeFunction,
            'Age': age
            }

    features = pd.DataFrame(data, index=[0])
    return features

def show_results1():
    st.subheader('User Input Client Features: ')
    st.write(df) 
    model_reg = pickle.load(open('model.pkl', 'rb'))
    prediction = model_reg.predict(df)    
    st.subheader('Cluster du Client : ')    
    st.write (prediction)

    if prediction == 0:
            st.write("Cluster 0 : Meilleurs Clients de la boite, avec une grande fréquence d’achats  et un chiffre d’affaires conséquent et des commandes récentes.")
            st.write("Un Client fidèle et representant Presque les 80% du chiffre d’affaire global, un client qui doit être orienté vers un traitement long term reposant sur des contrats de long durée et des offres de prix préférentielles.")
    elif prediction == 1:
        st.write("Cluster 1 : Clients avec des commandes anciènnes et qui ne commande plus assez, avec une fréquence et un chiffre d’affaire moyen.")
        st.write("Clients qui ont peut être cherché ailleurs pour leur commandes récente, l’objectif de l’equipe commerciale serais de les recontacter et savoir les causes de leurs changement, et par la suite essayer de les reconquerir avec des offres concurrentielles.")

def show_results2():

    st.subheader('User Input Patients Features: ')
    st.write(df_reg) 
    model_reg = pickle.load(open('model.pkl', 'rb'))
    prediction = model_reg.predict(df)
    prediction = pd.DataFrame(prediction)
    st.subheader('Clusters des Clients : ') 
    df_reg['Resultats']=prediction.values

if option=='input parameters directly':
    st.sidebar.header('User Input Parameters')
    df = user_input_features()
    show_results1()
    
elif option=='Load a file of data':
    uploaded_file = st.file_uploader("Choose a patients datafile with Columns as follows :(Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)")
    if uploaded_file is not None:
        df1 = pd.read_csv(uploaded_file)
        df = df1.iloc[1:,[0,1,2,3,4,5,6,7]].values
        #df_reg = pd.DataFrame({
        #'Pregnancies': df[:,0],
        #'Glucose': df[:,1],
        #'BloodPressure': df[:,2],
        #'SkinThickness': df[:,3],
        #'Insulin': df[:,4],
        #'BMI': df[:,5],
        #'DiabetesPedigreeFunction': df[:,6],
        #'Age': df[:,7]
        #})
        df_reg= pd.DataFrame(list(zip(
        df[:,0],
        df[:,1],
        df[:,2],
        df[:,3],
        df[:,4],
        df[:,5],
        df[:,6],
        df[:,7]
        )))
        show_results2()
    
