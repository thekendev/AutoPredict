from flask import Flask, render_template, request
import os
import numpy as np
from Autopredictor.src.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a Flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 




from flask import Flask, render_template, request
import os
import numpy as np
from Autopredictor.src.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a Flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

from flask import Flask, render_template, request
import os
import numpy as np
from Autopredictor.src.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a Flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            model = str(request.form['Model'])
            year = int(request.form['Year'])
            price = float(request.form['Price'])
            transmission = request.form['Transmission']
            mileage = float(request.form['Mileage'])
            fuel_type = request.form['FuelType']
            tax = float(request.form['Tax'])
            mpg = float(request.form['MPG'])
            engine_size = float(request.form['EngineSize'])
            manufacturer = str(request.form['Manufacturer'])

            data = [model, year, price, transmission, mileage, fuel_type, tax, mpg, engine_size, manufacturer]
            print("Data received from form:", data)  # Debugging line
            # Converting the data to a numpy array
            data = np.array(data).reshape(1, 10)
            print("Data reshaped for prediction:", data)  # Debugging line

            # Assuming you have a PredictionPipeline class to handle the prediction
            obj = PredictionPipeline()
            predict = obj.predict(data)
            print("Prediction result:", predict)  # Debugging line

            # Ensure the prediction result is correctly formatted
            if not isinstance(predict, (list, np.ndarray)) or len(predict) == 0:
                raise ValueError("Prediction result is not in the expected format.")

            # Convert the prediction to a categorical label
            prediction_label = convert_prediction_to_label(predict)

            return render_template('results.html', prediction=prediction_label)

        except Exception as e:
            # Log the exception and provide detailed feedback
            error_message = f"The Exception message is: {e}"
            print(error_message)
            return error_message

    else:
        return render_template('index.html')

def convert_prediction_to_label(prediction):
    # Assuming prediction is an array-like object with the predicted class index
    # Replace the following mapping with your actual class labels
    class_labels = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    return class_labels.get(prediction[0], "Unknown")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

