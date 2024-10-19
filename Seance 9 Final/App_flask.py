from flask import Flask, request,jsonify
import pickle

App = Flask(__name__)
model = pickle.load(ope("classifier_XGBC_final.pkl", "rb"))


@app.route('/')
def home():
    return 'Welcome to care proicing solution API'

if __name__=="__main__":
    App_flask.py(debug=True)

