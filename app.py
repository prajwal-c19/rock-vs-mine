from flask import Flask, render_template, request
import numpy as np
import pickle
import os  # Import os module

app = Flask(__name__)

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_text = request.form["features"]
        input_values = [float(x) for x in input_text.replace(",", " ").split()]
        
        if len(input_values) != 60:
            return render_template("index.html", prediction="Error: Enter exactly 60 values.")

        input_array = np.array(input_values).reshape(1, -1)
        input_array = scaler.transform(input_array)  # Scale input before prediction

        prediction = model.predict(input_array)[0]
        result = "Rock" if prediction == "R" else "Mine"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Correctly setting the port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)





