# 📊 Monitoring System Documentation

## Overview

Hệ thống monitoring hoàn chỉnh cho Logistics KPI Prediction Model với 4 components chính:

1. **PredictionLogger** - Log tất cả predictions
2. **PerformanceMonitor** - Theo dõi model performance
3. **DataDriftDetector** - Phát hiện data drift
4. **ModelHealthChecker** - Kiểm tra system health

---

## 🚀 Quick Start

### 1. Sử dụng qua API

**Xem prediction statistics (24h):**

```bash
curl http://localhost:8000/monitoring/predictions?hours=24
```

**Xem performance metrics:**

```bash
curl http://localhost:8000/monitoring/performance?last_n=10
```

**Detailed health check:**

```bash
curl http://localhost:8000/monitoring/health
```

**Evaluate model với validation data:**

```bash
curl -X POST http://localhost:8000/monitoring/evaluate \
  -F "file=@validation_data.csv"
```

### 2. Sử dụng trực tiếp trong Python

```python
from monitoring import (
    PredictionLogger,
    PerformanceMonitor,
    DataDriftDetector,
    ModelHealthChecker,
    run_monitoring_report
)

# 1. Log predictions
pred_logger = PredictionLogger()
pred_logger.log_prediction(
    item_data={'item_id': 'ABC123', 'category': 'Electronics', ...},
    prediction=0.8245,
    confidence='high',
    response_time=0.085,
    model_version='Ridge_Regression_v1.0',
    features_count=43
)

# 2. Get statistics
stats = pred_logger.get_statistics(hours=24)
print(f"Total predictions: {stats['total_predictions']}")
print(f"Average KPI: {stats['avg_kpi']:.4f}")

# 3. Evaluate performance
perf_monitor = PerformanceMonitor()
metrics = perf_monitor.evaluate_model(
    y_true=y_test,
    y_pred=predictions,
    dataset_name="test_set"
)
print(f"R²: {metrics['r2_score']:.4f}")

# 4. Check for data drift
drift_detector = DataDriftDetector()
drift_detector.set_reference_data(training_data)
drift_results = drift_detector.detect_drift(production_data)
print(f"Drift detected in {len(drift_results['drifted_features'])} features")

# 5. Health check
health_checker = ModelHealthChecker()
health = health_checker.check_health()
print(f"System status: {health['overall_status']}")

# 6. Generate comprehensive report
run_monitoring_report(hours=24)
```

---

## 📋 Components Details

### 1. PredictionLogger

**Purpose:** Log tất cả predictions để audit và analysis

**Files created:**

- `predictions_history.csv` - Chi tiết từng prediction

**Key methods:**

```python
# Log một prediction
pred_logger.log_prediction(
    item_data=dict,          # Input data
    prediction=float,        # Predicted KPI
    confidence=str,          # 'high', 'medium', 'low'
    response_time=float,     # Seconds
    model_version=str,       # Model name
    features_count=int       # Number of features
)

# Get recent predictions
recent = pred_logger.get_recent_predictions(hours=24)

# Get statistics
stats = pred_logger.get_statistics(hours=24)
```

**Output statistics:**

- Total predictions count
- Average/std/min/max KPI scores
- Response time metrics
- Predictions by category

### 2. PerformanceMonitor

**Purpose:** Monitor model performance over time

**Files created:**

- `performance_metrics.json` - Performance evaluations history

**Key methods:**

```python
# Evaluate model
metrics = perf_monitor.evaluate_model(
    y_true=np.array,      # Ground truth
    y_pred=np.array,      # Predictions
    dataset_name=str      # Dataset identifier
)

# Get history
history = perf_monitor.get_performance_history(last_n=10)
```

**Thresholds & Alerts:**

- R² < 0.95 → Warning
- RMSE > 0.01 → Warning
- MAE > 0.01 → Warning

**Metrics tracked:**

- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Sample count
- Timestamp

### 3. DataDriftDetector

**Purpose:** Detect distribution changes in production data

**Key methods:**

```python
# Set reference (training) data
drift_detector.set_reference_data(training_df)

# Detect drift in production data
drift_results = drift_detector.detect_drift(production_df)

# Get feature statistics
stats = drift_detector.get_feature_statistics(data)
```

**How it works:**

- Uses Kolmogorov-Smirnov test for each numeric feature
- Compares production vs training distributions
- p-value < 0.05 → Drift detected

**Output:**

```python
{
    'timestamp': '2024-11-18T...',
    'production_samples': 1000,
    'reference_samples': 3204,
    'features_analyzed': 43,
    'drifted_features': ['stock_level', 'daily_demand'],
    'drift_percentage': 4.65,
    'drift_scores': {
        'stock_level': {
            'ks_statistic': 0.1234,
            'p_value': 0.0123,
            'drifted': True
        },
        ...
    }
}
```

**Alerts:**

- > 20% features drifted → Critical (retrain recommended)
- > 0% features drifted → Warning (monitor closely)

### 4. ModelHealthChecker

**Purpose:** Comprehensive system health check

**Key methods:**

```python
health = health_checker.check_health()
```

**Checks performed:**

1. **Model Files** - Verify .pkl files exist
2. **Recent Activity** - Count predictions in last hour
3. **Response Time** - Check avg/max response times
4. **Model Performance** - Verify R² score

**Health statuses:**

- `healthy` - All checks pass
- `degraded` - Some warnings
- `unhealthy` - Critical issues

**Output:**

```python
{
    'timestamp': '2024-11-18T...',
    'overall_status': 'healthy',
    'checks': {
        'model_files': {
            'status': 'pass',
            'message': 'Model files found'
        },
        'recent_activity': {
            'status': 'pass',
            'predictions_last_hour': 150,
            'message': '150 predictions in last hour'
        },
        'response_time': {
            'status': 'pass',
            'avg_ms': 85.5,
            'max_ms': 150.0,
            'message': 'Avg: 86ms, Max: 150ms'
        },
        'model_performance': {
            'status': 'pass',
            'r2_score': 0.9999,
            'rmse': 0.0003,
            'message': 'R²=0.9999'
        }
    }
}
```

---

## 🎯 Use Cases

### Case 1: Daily Monitoring Report

```python
from monitoring import run_monitoring_report

# Generate daily report
run_monitoring_report(hours=24)
```

**Output:**

```
================================================================================
📊 MONITORING REPORT - Last 24 Hours
================================================================================

📝 Prediction Statistics:
   Total predictions: 1,245
   Average KPI: 0.7856
   KPI range: [0.1234, 0.9876]
   Avg response time: 92.45ms
   Max response time: 250.12ms

   Predictions by category:
      Electronics: 456
      Furniture: 321
      Clothing: 268
      Food: 200

📈 Recent Performance Evaluations:
   1. 2024-11-18 10:30:00
      R²=0.999900, RMSE=0.000300, MAE=0.000200
   ...

🏥 System Health: HEALTHY
   ✅ model_files: Model files found
   ✅ recent_activity: 150 predictions in last hour
   ✅ response_time: Avg: 86ms, Max: 150ms
   ✅ model_performance: R²=0.9999
```

### Case 2: Weekly Performance Evaluation

```python
import pandas as pd
from monitoring import PerformanceMonitor

# Load validation data
val_data = pd.read_csv('validation_set.csv')
y_true = val_data['kpi_score']

# Load model and predict
from predict import predict_kpi
predictions = predict_kpi(val_data)

# Evaluate
monitor = PerformanceMonitor()
metrics = monitor.evaluate_model(y_true, predictions, "weekly_validation")

# Check for alerts
if metrics['alerts']:
    print("⚠️ ALERTS:")
    for alert in metrics['alerts']:
        print(f"  {alert}")
    print("\n🔄 Consider retraining the model!")
```

### Case 3: Detect Data Drift

```python
import pandas as pd
from monitoring import DataDriftDetector

# Load training data (reference)
train_data = pd.read_csv('data/logistics_dataset.csv')

# Load production data (last week)
prod_data = pd.read_csv('production_data_week.csv')

# Detect drift
detector = DataDriftDetector()
detector.set_reference_data(train_data)
drift_results = detector.detect_drift(prod_data)

# Check results
if drift_results['drift_percentage'] > 20:
    print(f"🚨 CRITICAL: {drift_results['drift_percentage']:.1f}% features drifted!")
    print(f"Drifted features: {drift_results['drifted_features']}")
    print("📧 Send alert to data science team")
    print("🔄 Schedule model retraining")
elif drift_results['drift_percentage'] > 0:
    print(f"⚠️ Warning: {drift_results['drift_percentage']:.1f}% features drifted")
    print("Continue monitoring...")
else:
    print("✅ No significant drift detected")
```

### Case 4: Real-time Health Monitoring

```bash
# Create monitoring script (monitor_health.sh)
#!/bin/bash
while true; do
    echo "=== Health Check $(date) ==="
    curl -s http://localhost:8000/monitoring/health | jq '.health_status.overall_status'
    sleep 300  # Check every 5 minutes
done
```

---

## 📈 Dashboards & Visualization

### Streamlit Dashboard Integration

Monitoring data được tự động integrate vào dashboard:

**Access:** http://localhost:8501

**Features:**

- Real-time prediction stats
- Performance trends charts
- Health status indicators
- Drift detection alerts

### Grafana Integration (Optional)

```python
# Export metrics to Prometheus format
from monitoring import PredictionLogger

logger = PredictionLogger()
stats = logger.get_statistics(hours=1)

# Prometheus metrics
print(f"predictions_total {{period='1h'}} {stats['total_predictions']}")
print(f"kpi_average {{period='1h'}} {stats['avg_kpi']}")
print(f"response_time_ms {{period='1h'}} {stats['avg_response_time_ms']}")
```

---

## 🔔 Alerting Setup

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText
from monitoring import PerformanceMonitor

def send_alert_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@company.com'
    msg['To'] = 'datascience@company.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('user', 'password')
        server.send_message(msg)

# Check performance and alert
monitor = PerformanceMonitor()
history = monitor.get_performance_history(last_n=1)

if history and history[-1]['alerts']:
    send_alert_email(
        subject="⚠️ Model Performance Alert",
        body=f"Alerts: {history[-1]['alerts']}"
    )
```

### Slack Integration

```python
import requests
from monitoring import ModelHealthChecker

SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

def send_slack_alert(message):
    requests.post(SLACK_WEBHOOK, json={"text": message})

# Daily health check
checker = ModelHealthChecker()
health = checker.check_health()

if health['overall_status'] != 'healthy':
    send_slack_alert(
        f"⚠️ Model health status: {health['overall_status']}\n"
        f"Check details: http://localhost:8000/monitoring/health"
    )
```

---

## 🔄 Maintenance Schedule

| Task                       | Frequency | Command                                                                               |
| -------------------------- | --------- | ------------------------------------------------------------------------------------- |
| Generate monitoring report | Daily     | `python -c "from monitoring import run_monitoring_report; run_monitoring_report(24)"` |
| Check for drift            | Weekly    | See Case 3 above                                                                      |
| Evaluate on validation set | Weekly    | See Case 2 above                                                                      |
| Review performance metrics | Weekly    | `curl http://localhost:8000/monitoring/performance`                                   |
| Health check               | Hourly    | `curl http://localhost:8000/monitoring/health`                                        |
| Archive old logs           | Monthly   | Move `.log` and `.csv` files to archive folder                                        |

---

## 📊 Monitoring Files

All monitoring data is stored locally:

```
log_model/
├── monitoring_logs.log          # System logs
├── predictions_history.csv      # All predictions
├── performance_metrics.json     # Performance evaluations
├── api_logs.log                 # API request logs
└── monitoring.py                # Monitoring system code
```

**File rotation:**

```bash
# Rotate logs monthly (add to cron)
0 0 1 * * cd /path/to/log_model && \
  mv monitoring_logs.log monitoring_logs_$(date +\%Y\%m).log && \
  mv predictions_history.csv predictions_history_$(date +\%Y\%m).csv
```

---

## 🎯 Best Practices

### 1. Regular Evaluation

```python
# Evaluate weekly with held-out validation set
metrics = perf_monitor.evaluate_model(y_val, y_pred_val, "weekly_validation")
```

### 2. Monitor Drift Continuously

```python
# Check drift monthly
drift_results = detector.detect_drift(monthly_production_data)
if drift_results['drift_percentage'] > 20:
    trigger_retraining_pipeline()
```

### 3. Set Up Automated Alerts

```python
# Check health every hour
if health['overall_status'] != 'healthy':
    send_alert_to_team()
```

### 4. Track Response Times

```python
# Alert if response time > 1s
if stats['avg_response_time_ms'] > 1000:
    investigate_performance_issue()
```

### 5. Archive Historical Data

```bash
# Monthly archival
tar -czf archive_$(date +%Y%m).tar.gz *.log *.csv *.json
mv archive_*.tar.gz archives/
```

---

## 🔧 Troubleshooting

**Problem: monitoring_logs.log file too large**

```bash
# Truncate log file
echo "" > monitoring_logs.log

# Or set up rotation
logrotate -f logrotate.conf
```

**Problem: predictions_history.csv growing too large**

```python
# Keep only last 30 days
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('predictions_history.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
cutoff = datetime.now() - timedelta(days=30)
df = df[df['timestamp'] >= cutoff]
df.to_csv('predictions_history.csv', index=False)
```

**Problem: Performance evaluation slow**

```python
# Use sampling for large datasets
sample_size = min(10000, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)
y_true_sample = y_true[sample_indices]
y_pred_sample = y_pred[sample_indices]
```

---

## 📚 Additional Resources

- **Main Documentation:** README.md
- **Deployment Guide:** DEPLOYMENT_GUIDE.md
- **API Documentation:** http://localhost:8000/docs
- **Project Report:** PROJECT_REPORT.md

---

**✅ Monitoring system sẵn sàng! Track model performance 24/7!**

_Last Updated: November 18, 2025_
