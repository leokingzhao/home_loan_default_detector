# -*- coding: utf-8 -*-

import PIL
from PIL import Image
# from streamlit_shap import st_shap
import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder


plt.style.use('default')

st.set_page_config(
    page_title='Real-Time Fraud Detection',
    page_icon='👁️',
    layout='wide'
)

# dashboard title
# st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>Real-Time Default Detection</h1>", unsafe_allow_html=True)


# side-bar
def user_input_features():

    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⇣')


    feature1 = st.sidebar.slider('EXT_SOURCE_2', 0.000000001, 1.0, 0.5)
    feature2 = st.sidebar.slider('EXT_SOURCE_3', 0.000000001, 1.0, 0.5)
    feature3 = st.sidebar.slider('DAYS_REGISTRATION', -100000.00, 0.0, 0.0)
    feature4 = st.sidebar.slider('DAYS_LAST_PHONE_CHANGE', -30000.00, 0.0, 0.0)

    feature5 = st.sidebar.selectbox('NAME_CONTRACT_TYPE', ('Cash loans', 'Revolving loans'))
    feature6 = st.sidebar.selectbox('CODE_GENDER', ('F', 'M', 'XNA'))
    feature7 = st.sidebar.selectbox('FLAG_OWN_CAR', ('N', 'Y'))
    feature8 = st.sidebar.selectbox('NAME_INCOME_TYPE', ('Commercial associate', 'Working', 'Pensioner', 'State servant',
                                                 'Businessman', 'Unemployed', 'Student', 'Maternity leave'))
    feature9 = st.sidebar.selectbox('NAME_EDUCATION_TYPE', ('Higher education', 'Secondary / secondary special', 'Incomplete higher',
                                                    'Lower secondary', 'Academic degree'))


    output = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]
    return output

features = ['EXT_SOURCE_2','EXT_SOURCE_3','DAYS_REGISTRATION','DAYS_LAST_PHONE_CHANGE',
            'NAME_CONTRACT_TYPE','CODE_GENDER', 'FLAG_OWN_CAR','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
outputdf = user_input_features()

# understand the dataset
df = pd.read_csv('data_new.csv')

st.title('Dataset')
if st.button('View some random data'):
    st.write(df.sample(5))

st.write(
    f'The dataset is trained on LinearRegression with totally length of: {len(df)}. 0️⃣ means its a real loan, 1️⃣ means its a default loan. data is unbalanced (not⚖️)')

unbalancedf = pd.DataFrame(df.TARGET.value_counts())
st.write(unbalancedf)

# 需要一个count plot
placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

df2 = df[['TARGET', 'NAME_CONTRACT_TYPE']]
df3 = df[['TARGET', 'CODE_GENDER']]
df4 = df[['TARGET', 'FLAG_OWN_CAR']]
df5 = df[['TARGET', 'NAME_INCOME_TYPE']]
df6 = df[['TARGET', 'NAME_EDUCATION_TYPE']]


with placeholder1.container():
    f1, f2, f3, = st.columns(3)

    with f1:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df['TARGET'] == 1]['EXT_SOURCE_2'], label='Default', kde=True)
        sns.histplot(df[df['TARGET'] == 0]['EXT_SOURCE_2'], label='Real', kde=True)
        plt.xlabel('EXT_SOURCE_2')
        plt.title('EXT_SOURCE_2')
        plt.legend()
        st.pyplot(use_container_width=True)

    with f2:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df['TARGET'] == 1]['EXT_SOURCE_3'], label='Default', kde=True)
        sns.histplot(df[df['TARGET'] == 0]['EXT_SOURCE_3'], label='Real', kde=True)
        plt.xlabel('EXT_SOURCE_3')
        plt.title('EXT_SOURCE_3')
        plt.legend()
        st.pyplot(use_container_width=True)

    with f3:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df['TARGET'] == 1]['DAYS_REGISTRATION'], label='Default', kde=True)
        sns.histplot(df[df['TARGET'] == 0]['DAYS_REGISTRATION'], label='Real', kde=True)
        plt.xlabel('DAYS_REGISTRATION')
        plt.title('DAYS_REGISTRATION')
        plt.legend()
        st.pyplot(use_container_width=True)

with placeholder2.container():
    f1, f2, f3 = st.columns(3)

    with f1:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df['TARGET'] == 1]['DAYS_LAST_PHONE_CHANGE'], label='Default', kde=True)
        sns.histplot(df[df['TARGET'] == 0]['DAYS_LAST_PHONE_CHANGE'], label='Real', kde=True)
        plt.xlabel('DAYS_LAST_PHONE_CHANGE')
        plt.title('DAYS_LAST_PHONE_CHANGE')
        plt.legend()
        st.pyplot(use_container_width=True)

    with f2:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df2, x='NAME_CONTRACT_TYPE', hue='TARGET', palette='viridis')
        plt.title('CONTRACT_TYPE')
        st.pyplot(use_container_width=True)

    with f3:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df3, x='CODE_GENDER', hue='TARGET', palette='viridis')
        plt.title('GENDER')
        st.pyplot(use_container_width=True)

with placeholder3.container():
    f1, f2, f3 = st.columns(3)

    with f1:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df4, x='FLAG_OWN_CAR', hue='TARGET', palette='viridis')
        plt.title('OWN_CAR')
        st.pyplot(use_container_width=True)

    with f2:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df5, x='NAME_INCOME_TYPE', hue='TARGET', palette='viridis')
        plt.title('INCOME_TYPE')
        st.pyplot(use_container_width=True)

    with f3:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df6, x='NAME_EDUCATION_TYPE', hue='TARGET', palette='viridis')
        plt.title('EDUCATION_TYPE')
        st.pyplot(use_container_width=True)


st.title('SHAP Value')
#image1 = Image.open('shap_plot.png')
shapdatadf = pd.read_csv(r'shapdatadf.csv').drop('Unnamed: 0',axis=1)
shapvaluedf = pd.read_csv(r'shapvaluedf.csv').drop('Unnamed: 0',axis=1)
catmodel = CatBoostClassifier()
catmodel.load_model("default_detector_catboost")
outputdf = pd.DataFrame([outputdf], columns=features)
X_test = pd.read_csv(r'X_test.csv').drop('Unnamed: 0',axis=1)

placeholder4 = st.empty()
with placeholder4.container():
    f1, f2 = st.columns(2)

    with f1:
        st.subheader('Summary plot')
        st.write('👈 class 0: Real')
        st.write('👉 class 1: Default')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        explainer = shap.Explainer(catmodel)
        shap_values = explainer(X_test)
        
        # Plot the SHAP values
        
        explanation = shap.Explanation(values=shap_values, data=X_test, feature_names=features)
        
        # Create the beeswarm plot using the Explanation object
        shap.plots.beeswarm(explanation)
        st.pyplot()

    
    with f2:
        st.subheader('Dependence plot for features')
        cf = st.selectbox("Choose a feature", (shapvaluedf.columns))

        fig = px.scatter(x=shapdatadf[cf],
                         y=shapvaluedf[cf],
                         color=shapdatadf[cf],
                         color_continuous_scale=['blue', 'red'],
                         labels={'x': 'Original value', 'y': 'shap value'})
        st.write(fig)



st.title('Make predictions in real time')


# st.write('User input parameters below ⬇️')
# st.write(outputdf)


p1 = catmodel.predict(outputdf)[0]
p2 = catmodel.predict_proba(outputdf)

placeholder5 = st.empty()
with placeholder5.container():
    f1, f2 = st.columns(2)
    with f1:
        st.write('User input parameters below ⬇️')
        st.write(outputdf)
        st.write(f'Predicted class: {p1}')
        st.write('Predicted cla和ss Probability')
        st.write('0️⃣ means its a Good Loan, 1️⃣ means its a Default Loan')
        st.write(p2)
    with f2:
        
        # st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot()
