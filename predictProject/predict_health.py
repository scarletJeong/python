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
    # 모델 및 스케일러 로드
    scaler = joblib.load("scaler.pkl")
    models = {
        "diabetes": joblib.load("diabetes_model.pkl"),
        "hypertension": joblib.load("hypertension_model.pkl"),
        "obesity": joblib.load("obesity_model.pkl")
    }

    data = request.json
    input_data = np.array([[data['gender'], data['sbp'], data['dbp'], data['fasting_blood_sugar'], data['waist'], data['weight'], data['bmi']]])
    input_scaled = scaler.transform(input_data)  # 스케일링
    
    predictions = {}
    for disease, model in models.items():
        pred = model.predict(input_scaled)
        predictions[disease] = int(pred[0])  # 예측 결과
    
    return jsonify(predictions)

@app.route('/funtionalFoodRecommend', methods=['POST'])
def funtionalFoodRecommend():
    # JSON 데이터 출력
    data = request.json
    print("Raw Data:", request.data)
    print("Request Method:", request.method)
    print("Received JSON data:", data)

    # 모델 및 스케일러 로드
    scaler = joblib.load("funtionalfood_scaler.pkl")
    models = {
        "functional food recommend": joblib.load("functional_food_model.pkl"),
        
    }

    data = request.json
    input_data = np.array([[ data['sbp'], data['dbp'], data['gender'],data['fasting_blood_sugar'],  data['weight'], data['waist'], data['hypertension'], data['diabetes'], data['obesity'], data['bmi']]])
    input_scaled = scaler.transform(input_data)  # 스케일링
    
    predictions = {}
    for disease, model in models.items():
        pred = model.predict(input_scaled)
        predictions[disease] = int(pred[0])  # 예측 결과
    
    return jsonify(predictions)





@app.route('/test', methods=['get'])
def test():
    print("Hello !!!!!")
    return jsonify("Hello !!!!")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
