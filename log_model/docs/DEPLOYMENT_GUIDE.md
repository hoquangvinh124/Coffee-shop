# 🚀 DEPLOYMENT GUIDE - Logistics KPI Prediction System

## 📋 Bảng Nội Dung

- [Option 1: Local Development](#option-1-local-development)
- [Option 2: Docker Deployment](#option-2-docker-deployment)
- [Option 3: Cloud Deployment](#option-3-cloud-deployment)
- [Testing & Validation](#testing--validation)
- [Monitoring & Maintenance](#monitoring--maintenance)

---

## Option 1: Local Development

### 🔧 Cài Đặt Cơ Bản

```bash
# 1. Clone hoặc download project
cd log_model

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Kiểm tra model đã có
ls models/
# Phải thấy: Ridge_Regression_*.pkl, scaler_*.pkl, encoders_*.pkl
```

### 🏃 Chạy Ứng Dụng

#### A. FastAPI Backend (REST API)

```bash
# Chạy API server
python app.py

# Hoặc dùng uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Truy cập:**

- 🌐 API: http://localhost:8000
- 📚 Swagger Docs: http://localhost:8000/docs
- 📖 ReDoc: http://localhost:8000/redoc

**Test API:**

```bash
# Health check
curl http://localhost:8000/health

# Single prediction (POST request)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Electronics",
    "stock_level": 150,
    "reorder_point": 50,
    "reorder_frequency_days": 7,
    "lead_time_days": 3,
    "daily_demand": 15.5,
    "demand_std_dev": 3.2,
    "item_popularity_score": 0.75,
    "zone": "A",
    "picking_time_seconds": 45,
    "handling_cost_per_unit": 2.50,
    "unit_price": 99.99,
    "holding_cost_per_unit_day": 0.50,
    "stockout_count_last_month": 1,
    "order_fulfillment_rate": 0.95,
    "total_orders_last_month": 450,
    "turnover_ratio": 8.5,
    "layout_efficiency_score": 0.80,
    "last_restock_date": "2024-11-01",
    "forecasted_demand_next_7d": 110.0
  }'
```

#### B. Streamlit Dashboard (Interactive UI)

```bash
# Chạy dashboard
streamlit run dashboard.py

# Hoặc với custom port
streamlit run dashboard.py --server.port 8501
```

**Truy cập:**

- 🌐 Dashboard: http://localhost:8501

**Tính năng:**

- ✅ Single item prediction với form nhập liệu
- ✅ Batch prediction với CSV upload
- ✅ Model analytics và visualizations
- ✅ Feature importance insights

#### C. Command Line Prediction

```bash
# Dự đoán từ CSV
python predict.py

# Hoặc trong Python
python -c "from predict import batch_predict_and_save; batch_predict_and_save('data/logistics_dataset.csv')"
```

---

## Option 2: Docker Deployment

### 🐳 Build và Run với Docker

#### Single Container

```bash
# Build Docker image
docker build -t logistics-kpi-api .

# Run API container
docker run -d \
  --name logistics-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  logistics-kpi-api

# Run Dashboard container
docker run -d \
  --name logistics-dashboard \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  logistics-kpi-api \
  streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

#### Docker Compose (Recommended)

```bash
# Start all services (API + Dashboard)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

**Services:**

- 🔌 API: http://localhost:8000
- 📊 Dashboard: http://localhost:8501

**Useful Docker Commands:**

```bash
# Check running containers
docker ps

# View API logs
docker logs logistics-kpi-api

# View Dashboard logs
docker logs logistics-kpi-dashboard

# Enter container shell
docker exec -it logistics-kpi-api bash

# Remove all containers and images
docker-compose down --rmi all --volumes
```

---

## Option 3: Cloud Deployment

### ☁️ Azure Deployment

#### A. Azure Container Instances (Fastest)

```bash
# Login to Azure
az login

# Create resource group
az group create --name logistics-kpi-rg --location eastus

# Build and push to Azure Container Registry
az acr create --resource-group logistics-kpi-rg \
  --name logisticskpiacr --sku Basic

az acr build --registry logisticskpiacr \
  --image logistics-kpi-api:v1 .

# Deploy to Azure Container Instances
az container create \
  --resource-group logistics-kpi-rg \
  --name logistics-kpi-api \
  --image logisticskpiacr.azurecr.io/logistics-kpi-api:v1 \
  --dns-name-label logistics-kpi \
  --ports 8000
```

#### B. Azure App Service

```bash
# Create App Service Plan
az appservice plan create \
  --name logistics-kpi-plan \
  --resource-group logistics-kpi-rg \
  --sku B1 --is-linux

# Create Web App
az webapp create \
  --resource-group logistics-kpi-rg \
  --plan logistics-kpi-plan \
  --name logistics-kpi-api \
  --deployment-container-image-name logisticskpiacr.azurecr.io/logistics-kpi-api:v1
```

#### C. Azure Kubernetes Service (Production)

```bash
# Create AKS cluster
az aks create \
  --resource-group logistics-kpi-rg \
  --name logistics-kpi-cluster \
  --node-count 2 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group logistics-kpi-rg \
  --name logistics-kpi-cluster

# Deploy to Kubernetes
kubectl apply -f kubernetes/
```

### 🌍 AWS Deployment

#### A. AWS ECS (Elastic Container Service)

```bash
# Install AWS CLI and configure
aws configure

# Create ECR repository
aws ecr create-repository --repository-name logistics-kpi-api

# Build and push
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t logistics-kpi-api .
docker tag logistics-kpi-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/logistics-kpi-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/logistics-kpi-api:latest

# Create ECS cluster and service (via AWS Console or CloudFormation)
```

#### B. AWS Lambda (Serverless)

```bash
# Package application
pip install -t package -r requirements.txt
cd package && zip -r ../deployment.zip . && cd ..
zip -g deployment.zip app.py predict.py train_model.py

# Create Lambda function
aws lambda create-function \
  --function-name logistics-kpi-predictor \
  --runtime python3.10 \
  --role <lambda-execution-role-arn> \
  --handler app.handler \
  --zip-file fileb://deployment.zip
```

### 🔵 Google Cloud Platform

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/logistics-kpi-api

# Deploy to Cloud Run
gcloud run deploy logistics-kpi-api \
  --image gcr.io/<project-id>/logistics-kpi-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Testing & Validation

### 🧪 Chạy Unit Tests

```bash
# Run all tests
python test_model.py

# Run with verbose output
python test_model.py -v

# Run specific test class
python -m unittest test_model.TestPrediction

# Run with coverage (install coverage first: pip install coverage)
coverage run test_model.py
coverage report
coverage html  # Generate HTML report
```

**Expected Results:**

```
Tests run: 13
Successes: 13
Failures: 0
Errors: 0

✅ ALL TESTS PASSED!
```

### 📊 Validate Model Performance

```bash
# Re-train and validate
python train_model.py

# Check metrics in output:
# - R² Score should be > 0.99
# - RMSE should be < 0.001
# - MAE should be < 0.001
```

### 🔍 API Testing

**Using curl:**

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch_request.json

# CSV upload
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@data/logistics_dataset.csv" \
  -o predictions_output.csv
```

**Using Python:**

```python
import requests

# Test endpoint
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
data = {
    "category": "Electronics",
    "stock_level": 150,
    # ... other fields
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

## Monitoring & Maintenance

### 📈 Monitoring Setup

#### 1. Log Monitoring

```bash
# View API logs
tail -f api_logs.log

# View last 100 lines
tail -n 100 api_logs.log

# Search for errors
grep "ERROR" api_logs.log

# Count predictions
grep "Prediction completed" api_logs.log | wc -l
```

#### 2. Performance Metrics

**Create monitoring script** (`monitor.py`):

```python
import pandas as pd
import time
from predict import predict_single_item

# Sample item for testing
sample_item = {...}

# Measure response time
start = time.time()
prediction = predict_single_item(sample_item)
response_time = time.time() - start

print(f"Response Time: {response_time:.3f}s")
print(f"Prediction: {prediction:.4f}")

# Alert if response time > threshold
if response_time > 1.0:
    print("⚠️ ALERT: Response time exceeds threshold!")
```

Run monitoring:

```bash
# Run once
python monitor.py

# Run periodically (every 5 minutes)
while true; do python monitor.py; sleep 300; done
```

#### 3. Model Drift Detection

```python
# monitor_drift.py
import pandas as pd
from sklearn.metrics import r2_score

# Load new production data
new_data = pd.read_csv('production_data.csv')

# Get predictions and actual KPIs
predictions = model.predict(new_data_features)
actual = new_data['actual_kpi']

# Calculate current R²
current_r2 = r2_score(actual, predictions)

# Alert if R² drops below threshold
if current_r2 < 0.95:
    print(f"⚠️ ALERT: Model R² dropped to {current_r2:.4f}")
    print("Consider retraining the model!")
```

#### 4. Azure Application Insights (if deployed on Azure)

```bash
# Install SDK
pip install applicationinsights

# Add to app.py
from applicationinsights import TelemetryClient
tc = TelemetryClient('<instrumentation-key>')

# Track predictions
tc.track_event('Prediction', {'kpi_score': prediction})
tc.flush()
```

### 🔄 Retraining Schedule

**When to retrain:**

1. ⚠️ R² drops below 0.95
2. ⚠️ RMSE increases above 0.01
3. ⚠️ Significant feature distribution changes
4. ⚠️ New categories or data patterns emerge
5. 📅 Quarterly schedule (every 3 months)

**Retraining process:**

```bash
# 1. Backup current model
cp models/Ridge_Regression_*.pkl models/backup/

# 2. Prepare new data
# Add new data to data/logistics_dataset_new.csv

# 3. Retrain
python train_model.py

# 4. Validate new model
python test_model.py

# 5. Deploy new model (hot reload)
# API will automatically load latest model on restart
```

### 🛠️ Troubleshooting

**Problem: Model not loading**

```bash
# Check if model files exist
ls -la models/

# Check file permissions
chmod 644 models/*.pkl

# Verify Python can import
python -c "import joblib; joblib.load('models/Ridge_Regression_*.pkl')"
```

**Problem: API not responding**

```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Kill process and restart
kill -9 <PID>
python app.py
```

**Problem: Predictions too slow**

```bash
# Profile code
python -m cProfile -o profile.stats app.py

# Optimize:
# - Use batch predictions
# - Load model once (not per request)
# - Cache frequent predictions
```

---

## 🔐 Security Best Practices

### API Security

```python
# Add authentication (app.py)
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# Protect endpoints
@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_single(item: ItemFeatures):
    ...
```

### HTTPS Setup

```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with HTTPS
uvicorn app:app --ssl-keyfile key.pem --ssl-certfile cert.pem --port 443
```

### Environment Variables

```bash
# Create .env file
echo "MODEL_PATH=./models" > .env
echo "API_KEY=your-secret-key" >> .env
echo "LOG_LEVEL=INFO" >> .env

# Load in app
from dotenv import load_dotenv
load_dotenv()
```

---

## 📦 Backup & Recovery

### Backup Strategy

```bash
# Create backup script (backup.sh)
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"

mkdir -p $BACKUP_DIR
cp -r models/ $BACKUP_DIR/
cp -r data/ $BACKUP_DIR/
cp *.py $BACKUP_DIR/
cp requirements.txt $BACKUP_DIR/

echo "✅ Backup created: $BACKUP_DIR"
```

### Recovery Process

```bash
# Restore from backup
RESTORE_DATE="20251118_145155"
cp backups/$RESTORE_DATE/models/* models/
python test_model.py  # Validate
```

---

## 🎯 Performance Optimization

### API Optimization

```python
# app.py - Add caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(item_hash):
    # Prediction logic
    pass

# Add request batching
from collections import deque
request_queue = deque()

async def batch_processor():
    while True:
        if len(request_queue) >= 10:
            # Process batch
            pass
        await asyncio.sleep(0.1)
```

### Database for Predictions

```python
# Store predictions for analytics
import sqlite3

conn = sqlite3.connect('predictions.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (timestamp TEXT, item_id TEXT, kpi_score REAL)''')

# Log predictions
c.execute("INSERT INTO predictions VALUES (?, ?, ?)",
          (datetime.now().isoformat(), item_id, kpi_score))
conn.commit()
```

---

## ✅ Deployment Checklist

### Pre-Deployment

- [ ] All tests passing (`python test_model.py`)
- [ ] Model files present in `models/`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Environment variables configured
- [ ] SSL certificates ready (for HTTPS)
- [ ] Backup created

### Deployment

- [ ] API running (`http://localhost:8000/health` returns 200)
- [ ] Dashboard accessible (`http://localhost:8501`)
- [ ] Sample prediction successful
- [ ] Logs being written to `api_logs.log`
- [ ] Performance acceptable (<1s response time)

### Post-Deployment

- [ ] Monitor logs for 24 hours
- [ ] Verify predictions accuracy
- [ ] Set up automated monitoring
- [ ] Document API endpoints for users
- [ ] Schedule first maintenance review

---

## 📚 Additional Resources

- 📄 **README.md** - User guide
- 📋 **PROJECT_REPORT.md** - Technical details
- 📊 **exploratory_data_analysis.ipynb** - Data insights
- 🧪 **test_model.py** - Test suite
- 🐳 **docker-compose.yml** - Container orchestration

---

**🎉 Deployment hoàn tất! Model đã sẵn sàng production với R² = 99.99%**

_Last Updated: November 18, 2025_
