from flask import Flask,request,jsonify
import numpy as np
import joblib

#Create Flask Application
app=Flask(__name__)

#Load model and Scaler
model=joblib.load("house_price.pkl")
scaler=joblib.load("scaler.pkl")

#Home Route
@app.route("/")
def home():
    return "House Price Prediction Api is Running!"


#Prediction route
@app.route("/predict",methods=["POST"])
def predict():

   #Recieve data from frontend
   data=request.get_json()#json file is converted to dictionary

   #Extract input Features...........accessing values using keys
   features=[
    data["income"],
    data["age"],
    data["rooms"],
    data["bedrooms"],
    data["population"]
   ]

   #Convert into Numpy Array
   input_data=np.array(features).reshape(1,-1)

   #Scale the input
   input_scaled=scaler.transform(input_data)

   #Predict house price
   prediction=model.predict(input_scaled)

   #Return Prediction.....converting to json
   return jsonify({"predicted_price":float(prediction[0])})


#Run Flask Application
if __name__ == "__main__":
    app.run(debug=True)