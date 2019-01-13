from flask import Flask, request, jsonify
from sklearn.externals import joblib
import numpy as np
app = Flask(__name__)

# Load the model
MODEL = joblib.load('houseGrade-v1.0.pkl')
MODEL_LABELS = [1,2,3,4,5,6,7,8,9,10,11,12]

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    #'price','sqft_living',  'bedrooms', 'yr_built', 'yr_renovated'
    price = request.args.get('price')
    sqft = request.args.get('sqft_living')
    bedrooms = request.args.get('bedrooms')
    yearBuilt = request.args.get('yr_built')
    yearRenovated = request.args.get('yr_renovated')
    
    # Our model expects a list of records
    features = [[price, sqft, bedrooms, yearBuilt, yearRenovated]]


    # Use the model to predict the class
    label  = MODEL.predict(features)
    label_conf = MODEL.predict_proba(features)

    # Retrieve the iris name that is associated with the predicted class
    #label = MODEL_LABELS[label_index[0]]
    # Create and send a response to the API caller

    return jsonify(status='complete', label=label)



if __name__ == '__main__':
    app.run(debug=True)