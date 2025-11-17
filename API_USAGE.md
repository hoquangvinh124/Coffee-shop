# Coffee Shop API - Quick Usage Guide

## Start Server

```bash
python app.py
```

Server chạy tại: `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

## API Endpoints

### Overall System Forecasts

```bash
# Forecast N days (overall system)
POST /forecast?days=30

# Forecast date range (overall system)
GET /forecast/range?start_date=2025-01-01&end_date=2025-01-31
```

### Store Management

```bash
# List all stores
GET /stores

# Get top N stores by revenue
GET /stores/top/10

# Get specific store info
GET /stores/44
```

### Store Forecasts

```bash
# Forecast N days for a specific store
POST /stores/44/forecast?days=30

# Forecast date range for a specific store
GET /stores/44/forecast/range?start_date=2025-01-01&end_date=2025-01-31
```

## Examples

### 1. Get Top 5 Stores

**Request:**
```bash
curl http://localhost:8000/stores/top/5
```

**Response:**
```json
{
  "count": 5,
  "stores": [
    {
      "store_nbr": 44,
      "city": "Quito",
      "type": "A",
      "historical_avg_daily": 36869.09,
      "forecast_avg_daily": 55006.66,
      "growth_percent": 49.19
    },
    ...
  ]
}
```

### 2. Get Store Information

**Request:**
```bash
curl http://localhost:8000/stores/44
```

**Response:**
```json
{
  "store_nbr": 44,
  "city": "Quito",
  "state": "Pichincha",
  "type": "A",
  "cluster": 5,
  "historical_avg_daily": 36869.09,
  "forecast_avg_daily": 55006.66,
  "growth_percent": 49.19,
  "data_from": "2013-01-01",
  "data_to": "2017-08-15"
}
```

### 3. Forecast for Store

**Request:**
```bash
curl -X POST "http://localhost:8000/stores/44/forecast?days=30"
```

**Response:**
```json
{
  "store_nbr": 44,
  "city": "Quito",
  "type": "A",
  "forecast_start": "2017-08-16 00:00:00",
  "forecast_end": "2017-09-14 00:00:00",
  "forecast_days": 30,
  "historical_avg_daily": 36869.09,
  "forecast_avg_daily": 49611.58,
  "total_forecast": 1488347.41,
  "growth_percent": 34.56,
  "forecasts": [
    {
      "date": "2017-08-16",
      "forecast": 44359.20,
      "lower_bound": 33842.65,
      "upper_bound": 54826.98
    },
    ...
  ]
}
```

### 4. Forecast Date Range

**Request:**
```bash
curl "http://localhost:8000/stores/44/forecast/range?start_date=2017-09-01&end_date=2017-09-30"
```

**Response:**
```json
{
  "store_nbr": 44,
  "city": "Quito",
  "type": "A",
  "start_date": "2017-09-01",
  "end_date": "2017-09-30",
  "total_days": 30,
  "total_forecast": 1600467.98,
  "avg_daily_forecast": 53348.93,
  "min_daily": 38094.61,
  "max_daily": 74180.58,
  "forecasts": [...]
}
```

### 5. Overall System Forecast

**Request:**
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

**Response:**
```json
{
  "forecast_start": "2025-11-18",
  "forecast_end": "2025-11-24",
  "total_days": 7,
  "summary": {
    "avg_daily_forecast": 450123.45,
    "total_forecast": 3150864.15,
    "min_forecast": 420000.00,
    "max_forecast": 480000.00,
    "std_forecast": 18234.56
  },
  "forecasts": [...]
}
```

## Testing

Run automated tests:

```bash
python test_api.py
```

## Python Integration

```python
import requests

# Get store info
response = requests.get("http://localhost:8000/stores/44")
store_info = response.json()

# Forecast
response = requests.post("http://localhost:8000/stores/44/forecast?days=30")
forecast = response.json()

print(f"Total forecast: ${forecast['total_forecast']:,.2f}")
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad request (invalid parameters)
- **404**: Not found (invalid store number)
- **500**: Server error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Limits

- Max forecast days: 730 (2 years) for store-level
- Max forecast days: 365 (1 year) for overall system
- Valid store numbers: 1-54

## Documentation

- Full API docs: http://localhost:8000/docs (Swagger UI)
- Alternative docs: http://localhost:8000/redoc
- Detailed guide: [PREDICTION_GUIDE.md](PREDICTION_GUIDE.md)
- Project summary: [SUMMARY.md](SUMMARY.md)
