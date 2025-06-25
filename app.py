from flask import Flask
from flask import render_template, request
import pickle
import numpy as np
from os import path
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("home.html")


@app.route("/result", methods=["POST"])
def result_page():
    cntry = int(request.form["cntry"])
    b_s = int(request.form["b_s"])
    s_s = int(request.form["s_s"])
    w_s = int(request.form["w_s"])
    cntnt = int(request.form["cntnt"])

    model_path = path.join(app.root_path, "static", "model.pkl")
    with open(model_path, "rb") as f:
        lr_model = pickle.load(f)

    pred = lr_model.predict(np.array([[cntry, b_s, s_s, w_s, cntnt]]))[0]

    return render_template("result.html", prediction=pred)
