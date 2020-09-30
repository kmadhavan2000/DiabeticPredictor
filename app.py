import numpy as np
import tensorflow 
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model


app = Flask(__name__)
model = load_model('Diabetic_Predictor.h5')
st = pickle.load(open('ss1.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    preg = float(request.form['preg'])
    gluco = float(request.form['gluco'])
    bp = float(request.form['bp'])
    skinthck = float(request.form['skinthck'])
    insu = float(request.form['insu'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dbf'])
    age = float(request.form['age'])
    
    finalfeatures = st.transform(np.array([[preg,gluco,bp,skinthck,insu,bmi,dpf,age]]))
    prediction = model.predict(finalfeatures)
    return render_template('index.html', prediction_text='Diabetic is  $ {}'.format(round(prediction)))


if __name__ == "__main__":
    app.run(debug=True)
    
    
