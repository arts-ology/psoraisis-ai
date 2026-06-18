from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# load model, scaler, and top genes
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
#loading all the saved genes, the trained model, and scaler
with open("top_genes.pkl", "rb") as f:
    top_genes = pickle.load(f)

def get_risk_label(prob):
    if prob < 0.20:
        return "Low risk of psoriasis"
    elif prob < 0.40:
        return "Slightly elevated risk of psoriasis"
    elif prob < 0.60:
        return "Moderate risk of psoriasis — further testing recommended"
    elif prob < 0.80:
        return "High risk of psoriasis"
    else:
        return "Very high risk of psoriasis"

def score_patient(patient_data):
    patient_median = patient_data.median(axis=1).values[0]
    patient_centered = patient_data - patient_median
    patient_scaled = scaler.transform(patient_centered)
    prob = model.predict_proba(patient_scaled)[0][1]
    return float(prob)
#when the webpage sends a file like the patient data to analyze, it will run this 
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    df = pd.read_csv(file, index_col=0)
    
    #onlythe 300 genes
    patient = df[top_genes]
    
    #output the risk scores and labels 
    prob = score_patient(patient)
    label = get_risk_label(prob)
    
    return jsonify({
        "probability": round(prob * 100, 1),
        "label": label
    })

if __name__ == "__main__":
    app.run(debug=True)