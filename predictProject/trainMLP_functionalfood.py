# trainMLP_functionalfood.py

import pandas as pd
from sqlalchemy import create_engine # type: ignore
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib

# PostgreSQL 연결 설정
def load_data():
    engine = create_engine("postgresql+psycopg2://repurehc:repuredev1!@repurehc.chayw4we4abo.ap-northeast-2.rds.amazonaws.com:5432/purewithme")
    query = "SELECT gender, sbp, dbp, blood_sugar,waist, weight, bmi,diabetes,  hypertension, obesity, functional_food FROM disease_classification"
    df = pd.read_sql(query, engine)
    return df

# 데이터 로드 및 모델 학습
def train_model():
    data = load_data()
    X = data[['gender', 'sbp', 'dbp','blood_sugar', 'waist', 'weight', 'bmi','diabetes',  'hypertension', 'obesity']]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values)
    joblib.dump(scaler, "funtionalfood_scaler.pkl")
    
    # labels = {'diabetes': data['diabetes'], 'hypertension': data['hypertension'], 'obesity': data['obesity']}
    labels = {'functional_food': data['functional_food']}
    models = {}
    
    for label_name, y in labels.items():
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        #model = LogisticRegression()
        # model = LinearSVC(max_iter=10000, random_state=0)
        model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print(f"Classification report for {label_name}:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        models[label_name] = model
        joblib.dump(model, f"{label_name}_model.pkl")
    
    return models


# 모델 학습
train_model()
