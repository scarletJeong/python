from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 모델 및 스케일러 로드
scaler = joblib.load("scaler.pkl")
models = {
    "diabetes": joblib.load("diabetes_model.pkl"),
    "hypertension": joblib.load("hypertension_model.pkl"),
    "obesity": joblib.load("obesity_model.pkl")
}

# 예측 API 엔드포인트
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello !! Python Web Ready. </h1>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([[data['gender'], data['sbp'], data['dbp'], data['fasting_blood_sugar'], data['waist'], data['weight'], data['bmi']]])
    input_scaled = scaler.transform(input_data)  # 스케일링
    
    predictions = {}
    for disease, model in models.items():
        pred = model.predict(input_scaled)
        predictions[disease] = int(pred[0])  # 예측 결과
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
