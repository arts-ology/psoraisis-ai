from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# load model and top genes (no scaler needed!)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    df = pd.read_csv(file, index_col=0)
    patient = df[top_genes]
    prob = float(model.predict_proba(patient)[0][1])
    label = get_risk_label(prob)
    return jsonify({
        "probability": round(prob * 100, 3),
        "label": label
    })

if __name__ == "__main__":
    app.run(debug=True)