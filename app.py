from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model & scaler
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from textarea
        input_text = request.form["features"]
        
        # Convert input string into a list of floats
        input_values = [float(x) for x in input_text.replace(",", " ").split()]
        
        # Ensure exactly 60 values are entered
        if len(input_values) != 60:
            return render_template("index.html", prediction="Error: Enter exactly 60 values.")

        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)

        # Apply the same scaler used in training
        input_scaled = scaler.transform(input_array)

        # Get prediction
        prediction = model.predict(input_scaled)[0]

        # Convert output to "Rock" or "Mine"
        result = "Rock" if prediction == "R" else "Mine"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)




