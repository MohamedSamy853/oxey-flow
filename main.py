import pickle
import numpy as np
from flask import Flask , request, jsonify

app = Flask(__name__)

with open("model_pipr.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "hellpo from home page"

@app.route("/predict",methods=["POST"])
def predict():
    age = request.args.get("age")
    gender = request.args.get("gender")
    spo2 = request.args.get("spo2")
    pr = request.args.get("pr")
    nCov2 = request.args.get("nCoV2")
    data = np.array([age, gender, spo2, pr, nCov2]).astype(float).reshape(1 , -1)
    res = model.predict(data)
    return jsonify({"oxy_flow":str(res)})

if __name__ == '__main__':
    app.run(port=5000)
