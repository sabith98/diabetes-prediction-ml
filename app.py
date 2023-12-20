from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return '<h1>Welcome to diabetes prediction</h1>'

@app.route('/predictdata',methods=['POST'])
def predict_datapoint():
    if request.method=='POST':
        data = request.get_json()

        data=CustomData(
            Pregnancies=int(data['Pregnancies']),
            Glucose=int(data['Glucose']),
            BloodPressure=int(data['BloodPressure']),
            SkinThickness=int(data['SkinThickness']),
            Insulin=int(data['Insulin']),
            BMI=float(data['BMI']),
            DiabetesPedigreeFunction=float(data['DiabetesPedigreeFunction']),
            Age=int(data['Age'])
        )

        pred_df=data.get_data_as_data_frame()
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        result_str = "Patient has diabetes" if results[0]==1.0 else "Patient doesn't have diabetes"

        return result_str
    
if __name__=="__main__":
    app.run(host="0.0.0.0")