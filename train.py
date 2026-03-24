import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from mlflow.tracking import MlflowClient
experiment_name = "iris_classification1"

# [추가] MLflow 서버 설정 (환경 변수 활용)
# GitHub Actions 설정 시 Secrets에 넣을 주소임
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# [팁] Scikit-learn 자동 기록 켜기
mlflow.sklearn.autolog(log_models=False)

# 1. 데이터 로드 (실험 시작 전)
try:
    df = pd.read_csv("data/iris_local.csv")
    # 전처리 (수치형 데이터만 선택 및 결측치 제거)
    df = df.select_dtypes(include=['number']).dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    print("❌ 데이터를 찾을 수 없음. dvc pull 확인 필요.")
    exit(1)
    
#iris = load_iris()
#X, y = iris.data, iris.target
#X_train, X_test, y_train, y_test = train_test_split(
#   X, y, test_size=0.2, random_state=42, stratify=y

#)



# 플로우 시작
# [중요] 실험 결과를 담을 리스트 초기화
run_results = []    
param_list = [
    {"n_estimators": 50,  "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": None},
]

for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        mlflow.log_params(params)        # 딕셔너리로 한 번에!
        mlflow.log_metric("accuracy", acc)# 정확도 기록
        # 모델 저장 및 정보 가져오기 (artifact_path는 "model"로 지정)
        model_info = mlflow.sklearn.log_model(pipe, name="model")
        # [수정] 결과를 리스트에 저장 (나중에 비교하기 위함)
        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "model_uri": model_info.model_uri
        })        
        # 터미널에 결과 출력 (URI 포함)
        print(f"  {run_name}: {acc:.4f} | uri: {model_info.model_uri}")
# 실험 후 — 가장 좋은 모델 자동 선택
best = max(run_results, key=lambda x: x["accuracy"])
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# Model Registry에 등록
import mlflow
from mlflow.tracking import MlflowClient

registered = mlflow.register_model(
    model_uri=best["model_uri"],   # 이전 시간에 저장한 URI!
    name="iris_classifier"
)
print(f"✅ 등록 완료! Version: {registered.version}")

# Alias 설정 (production 표시)
client = MlflowClient()
client.set_registered_model_alias(
    name="iris_classifier",
    alias="production",
    version=registered.version
)
print(f"🚀 production alias → Version {registered.version}")



