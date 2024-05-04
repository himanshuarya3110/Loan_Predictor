from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
app = Flask(__name__)

decision = pickle.load(open('new_model.pkl','rb'))







@app.route('/')
def home():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def submit_form():
    # d = request.form.get('no_of_dependents')
    # You might want to handle or store this data differently.
    # print(type(data))
    # d =  int(request.form.get("no_of_dependents"))
    # print(type(d))
    # # e = request.form.get("education")
    # e = 1
    # # s =request.form.get("self_employed")
    # s = 0
    # i= int(request.form.get("income_annum"))
    # print(f"type of i is {type(i)}")
    # l =int(request.form.get("loan_amount"))
    # lt=int(request.form.get("loan_term"))
    # c=int(request.form.get("cibil_score"))
    # r =int(request.form.get("residential_assets_value"))
    # cv =int(request.form.get("commercial_assets_value"))
    # la=int(request.form.get("luxury_assets_value"))
    # b=int(request.form.get("bank_asset_value"))  
    # arr = np.array([[d,e,s,i,l,lt,c,r,cv,la,b]])
    # data_f = pd.DataFrame(arr)
    # print(arr)
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

    # Display the updated DataFrame with encoded columns
    # print(df_ml[['education', 'self_employed','loan_status']])



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


    # d = 
    # no_of_dependents =  3
    # education  =  0.000000
    # self_employed   =  0.000000
    # income_annum =  96000000
    # loan_amount = 199000000
    # loan_term = 15
    # cibil_score = 100
    # residential_assets_value = 4200000
    # commercial_assets_value  = 16200000
    # luxury_assets_value =  28500000
    # bank_asset_value = 6600000

    new_data  = pd.DataFrame({
    "no_of_dependents":[2],
    "education":[0.000000],
    "self_employed":[0.000000],
    "income_annum":[0],
    "loan_amount":[10300000],
    "loan_term":[10],
    "cibil_score":[636],
    "residential_assets_value":[1000000],
    "commercial_assets_value":[0],
    "luxury_assets_value":[6200000],
    "bank_asset_value":[3300000]

    })
    X_new = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']
    # scaler = StandardScaler()
    # print(result)
    new_data[X_new] = scaler.transform(new_data[X_new])
    # if result==1:
    #     return 'Approved'
    # else:
    #     return 'Rejected'
    # print(result)
    print('ooooooooooooooo')
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
    # print(new_data.at[0,'bank_asset_value'])
    # result = decision.predict(new_data)
    print(no_of_dependents)
    result = decision.predict([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])
    # print(new_data)
    if(result==0):
        return "{}".format('<h1> Approved</h1>')
    else:
        return "{}".format('<h1> Rejected </h1>')
    return "{}".format(result)
    # return "{}".format(result)
    # return "{}".format([[d,e,s,i,l,lt,c,r,cv,la,b]])


if __name__ == '__main__':
    app.run(debug=True)
