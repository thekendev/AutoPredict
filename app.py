from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd  # Import pandas
from Autopredictor.src.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a Flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            model = str(request.form['Model'])
            year = int(request.form['Year'])
            price = float(request.form['Price'])
            transmission = str(request.form['Transmission'])
            mileage = float(request.form['Mileage'])
            fuel_type = str(request.form['FuelType'])
            tax = float(request.form['Tax'])
            mpg = float(request.form['MPG'])
            engine_size = float(request.form['EngineSize'])
            manufacturer = str(request.form['Manufacturer'])

            data = [model, year, price, transmission, mileage, fuel_type, tax, mpg, engine_size, manufacturer]
            print("Data received from form:", data)  # Debugging line
            
            # Convert the data to a pandas DataFrame with correct column names
            columns = ['model', 'year', 'price', 'transmission', 'mileage', 'fueltype', 'tax', 'mpg', 'enginesize', 'manufacturer']
            data = pd.DataFrame([data], columns=columns)
            print("Data converted to DataFrame for prediction:", data)  # Debugging line

            # Assuming you have a PredictionPipeline class to handle the prediction
            obj = PredictionPipeline()
            predict = obj.predict(data)
            print("Prediction result:", predict)  # Debugging line

            # Convert the prediction to a string if it is not already
            prediction_result = str(predict[0]) if isinstance(predict, (list, np.ndarray)) else str(predict)

            return render_template('results.html', prediction=prediction_result)

        except Exception as e:
            # Log the exception and provide detailed feedback
            error_message = f"The Exception message is: {e}"
            print(error_message)
            return error_message

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
