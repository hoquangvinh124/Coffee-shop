"""
Quick test for the API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("="*80)
    print("TESTING COFFEE SHOP API")
    print("="*80)

    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # Test 2: Top stores
    print("\n2. Getting top 5 stores...")
    response = requests.get(f"{BASE_URL}/stores/top/5")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['count']} stores")
        for store in data['stores']:
            print(f"  Store {store['store_nbr']}: {store['city']} - ${store['historical_avg_daily']:,.0f}/day")

    # Test 3: Store info
    print("\n3. Getting store 44 info...")
    response = requests.get(f"{BASE_URL}/stores/44")
    if response.status_code == 200:
        info = response.json()
        print(f"Store {info['store_nbr']} - {info['city']} ({info['type']})")
        print(f"Historical avg: ${info['historical_avg_daily']:,.2f}")
        print(f"Forecast avg: ${info['forecast_avg_daily']:,.2f}")
        print(f"Growth: {info['growth_percent']}%")

    # Test 4: Forecast for store
    print("\n4. Forecasting 7 days for store 44...")
    response = requests.post(f"{BASE_URL}/stores/44/forecast?days=7")
    if response.status_code == 200:
        forecast = response.json()
        print(f"Total forecast: ${forecast['total_forecast']:,.2f}")
        print(f"Average daily: ${forecast['forecast_avg_daily']:,.2f}")
        print(f"Growth: {forecast['growth_percent']}%")
        print(f"\nFirst 3 days:")
        for day in forecast['forecasts'][:3]:
            print(f"  {day['date']}: ${day['forecast']:,.2f}")

    # Test 5: Overall system forecast
    print("\n5. Overall system forecast (7 days)...")
    response = requests.post(f"{BASE_URL}/forecast", json={"days": 7})
    if response.status_code == 200:
        forecast = response.json()
        print(f"Total forecast: ${forecast['summary']['total_forecast']:,.2f}")
        print(f"Average daily: ${forecast['summary']['avg_daily_forecast']:,.2f}")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure the server is running:")
        print("  python app.py")
