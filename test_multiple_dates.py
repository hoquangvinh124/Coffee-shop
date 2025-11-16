"""
Test predictions for multiple dates to see how model behaves
"""
import subprocess
import re

dates_to_test = [
    "2023-07-15",  # 2 weeks after training data
    "2023-08-01",  # 1 month after
    "2023-12-25",  # 6 months after
    "2024-01-01",  # 6 months after
    "2024-06-15",  # 1 year after
    "2025-01-01",  # 1.5 years after
    "2025-06-15",  # 2 years after
    "2026-01-01",  # 2.5 years after
]

print("=" * 80)
print("TESTING PREDICTIONS FOR MULTIPLE FUTURE DATES")
print("=" * 80)
print(f"\nTraining data ends: 2023-06-30")
print(f"Testing {len(dates_to_test)} dates from 2023-07-15 to 2026-01-01\n")
print("-" * 80)
print(f"{'Date':<15} {'Days After':<15} {'Predicted Revenue':<20}")
print("-" * 80)

from datetime import datetime

training_end = datetime(2023, 6, 30)

for date in dates_to_test:
    result = subprocess.run(
        ["python", "predict_future.py", date],
        capture_output=True,
        text=True
    )

    # Extract prediction from output
    for line in result.stdout.split('\n'):
        if '💰' in line or 'Doanh thu dự đoán' in line:
            # Extract dollar amount
            match = re.search(r'\$([0-9,]+\.\d{2})', line)
            if match:
                prediction = match.group(1)

                # Calculate days after training
                target_date = datetime.strptime(date, "%Y-%m-%d")
                days_after = (target_date - training_end).days

                print(f"{date:<15} {days_after:<15} ${prediction:<20}")
                break

print("-" * 80)
print("\n⚠️  OBSERVATIONS:")
print("- Model predictions for far future dates may not be reliable")
print("- Predictions assume revenue patterns continue similarly")
print("- For best accuracy, retrain model regularly with new data")
print()
