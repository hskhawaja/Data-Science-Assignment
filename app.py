from flask import Flask, render_template, url_for, request  
from sklearn.externals import joblib
from sklearn.preprocessing import scale
import numpy as np

app = Flask(__name__)    

@app.route('/')   
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])   
def predict():
    saved_model = joblib.load('ModelBalancedClasses.joblib') 
    
    if request.method == 'POST':
        cpi = request.form['cpi']
        cci = request.form['cci']
        data = scale(np.array([float(cpi), float(cci)]))
        my_prediction = saved_model.predict(data.reshape(1,-1))
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':  # Script executed directly (instead of via import)?
    app.run(debug=True)