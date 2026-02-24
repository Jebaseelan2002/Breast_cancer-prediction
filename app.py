from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    wr = float(request.form['worst_radius'])
    wp = float(request.form['worst_perimeter'])
    wa = float(request.form['worst_area'])
    wcp = float(request.form['worst_concave_points'])
    mcp = float(request.form['mean_concave_points'])

    # Arrange input
    input_data = np.array([[wr, wp, wa, wcp, mcp]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        result = "Benign (Non-Cancerous)"
    else:
        result = "Malignant (Cancerous)"

    confidence = round(np.max(probability) * 100, 2)

    return render_template(
        'index.html',
        prediction_text=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
