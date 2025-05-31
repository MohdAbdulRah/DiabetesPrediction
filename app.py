import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open("Models/model.pkl","rb"))
scaler = pickle.load(open("Models/scaler.pkl","rb"))



@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if(request.method == "POST"):
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure = int(request.form.get("BloodPressure"))
        SkinThickness = int(request.form.get("SkinThickness"))
        Insulin = int(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = int(request.form.get("Age"))
   
        new_dataScaled = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = model.predict(new_dataScaled)

        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")