from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
app = Flask(__name__)

filename = 'savedmodel.sav'
decision = pickle.load(open(filename,'rb'))
# decision.predict([[0.295621,0,0,1.617979,0.527139,	0.542970,	-0.858047,	-0.503257,	2.558270,	1.469203,	0.499509]])


@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict',methods=['POST','GET'])
def predict():
    df=pd.read_csv('loan_approval_dataset.csv')
    df_ml = df.copy()
    df_ml.drop(columns=['loan_id'], inplace=True)
    # Remove leading spaces from column names
    df_ml.rename(columns=lambda x: x.strip(), inplace=True)
    # print(df_ml.columns)
    # Display the updated DataFrame


    label_encoder = LabelEncoder()

    # Apply label encoding to the 'education' column
    df_ml['education'] = label_encoder.fit_transform(df_ml['education'])

    # Apply label encoding to the 'self_employed' column
    df_ml['self_employed'] = label_encoder.fit_transform(df_ml['self_employed'])

    # Apply label encoding to the 'loan_status' column
    df_ml['loan_status'] = label_encoder.fit_transform(df_ml['loan_status'])
    
    

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Define the feature columns (X) and target column (y)
    x = df_ml.drop(columns=['loan_status'])  # Drop 'loan_status' column to get feature columns
    y = df_ml['loan_status']  # Target variable

    # Select only the numerical columns for scaling (excluding 'loan_status')
    numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                        'bank_asset_value']

    # Apply scaling to the numerical columns
    x[numerical_columns] = scaler.fit_transform(x[numerical_columns])


    depd =  (request.form.get('no_of_dependents')) 
    edu =  (request.form.get('education'))
    if(edu=="Graduate"):
        edu = 0
    else:
        edu = 1
    empd =  (request.form.get('self_employed'))
    if(empd=="Yes"):
        empd = 1
    else:
        empd = 0
    income =  (request.form.get('income_annum'))
    loan_a =  (request.form.get('loan_amount'))
    loan_t =  (request.form.get('loan_term'))
    cibil_s =  (request.form.get('cibil_score'))
    rav =  (request.form.get('residential_assets_value'))
    cav =  (request.form.get('commercial_assets_value'))
    lav =  (request.form.get('luxury_assets_value'))
    bav =  (request.form.get('bank_asset_value'))

    
    new_data  = pd.DataFrame({
    "no_of_dependents":[depd],
    "education":[edu],
    "self_employed":[empd],
    "income_annum":[income],
    "loan_amount":[loan_a],
    "loan_term":[loan_t],
    "cibil_score":[cibil_s],
    "residential_assets_value":[rav],
    "commercial_assets_value":[cav],
    "luxury_assets_value":[lav],
    "bank_asset_value":[bav]

    })
    
    X_new = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

    new_data[X_new] = scaler.transform(new_data[X_new])

    no_of_dependents =  new_data.at[0,'no_of_dependents']
    education  = new_data.at[0,'education'] 
    self_employed   =  new_data.at[0,'self_employed']
    income_annum =  new_data.at[0,'income_annum']
    loan_amount = new_data.at[0,'loan_amount']
    loan_term = new_data.at[0,'loan_term']
    cibil_score = new_data.at[0,'cibil_score']
    residential_assets_value = new_data.at[0,'residential_assets_value']
    commercial_assets_value  = new_data.at[0,'commercial_assets_value']
    luxury_assets_value =  new_data.at[0,'luxury_assets_value']
    bank_asset_value = new_data.at[0,'bank_asset_value']
    
    
    result = decision.predict([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])[0]
    # result = decision.predict([[0.295621,0,0,1.617979,0.527139,	0.542970,	-0.858047,	-0.503257,	2.558270,	1.469203,	0.499509]])[0]
    # result = decision.predict([[-0.294102,	0,	0,	0.406510,	0.914208,	-0.508091,	0.029372,	0.388656,	1.327768,	0.733156,	1.299557]])[0]
    # print(no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value)
    # print(result)
    
    
    result = result.item()
    if(result == 0):
        result = "👌 Approved 👌"
    else:
        result = "❎ Rejected ❎"
    return render_template('predict.html',**locals())



if __name__ == '__main__':
    app.run(debug=True)
