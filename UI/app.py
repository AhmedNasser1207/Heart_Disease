import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
top_features = joblib.load('top_features.pkl')

original_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


st.title('Heart Disease Prediction App')
st.write('Enter patient data to predict the likelihood of heart disease.')

age = st.slider('Age', 20, 80, 50)
sex = st.selectbox('Sex', [0, 1]) # 1=Male, 0=Female
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
chol = st.slider('Cholesterol (chol)', 100, 600, 200)

user_input = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol}

placeholder_features = ['fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
for p in placeholder_features:
    user_input[p] = 0 

input_df = pd.DataFrame([user_input])
input_df = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)

X_scaled_cols = joblib.load('X_scaled_cols.pkl') # You should save this list too
input_df = input_df.reindex(columns=X_scaled_cols, fill_value=0)

input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

input_final = input_scaled_df[top_features]

if st.button('Predict'):
    prediction = model.predict(input_final)[0]
    prediction_proba = model.predict_proba(input_final)[0][1]

    if prediction == 1:
        st.error(f'High risk of heart disease. (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'Low risk of heart disease. (Probability: {prediction_proba:.2f})')