"""Inspect Raw Data.xlsx structure in detail"""
import pandas as pd
import numpy as np

file_path = '/home/user/Coffee-shop/Raw Data.xlsx'
xl = pd.ExcelFile(file_path)

print("="*80)
print("INSPECTING: Raw Data.xlsx")
print("="*80)

print(f"\nSheets: {xl.sheet_names}")

for sheet_name in xl.sheet_names:
    print("\n" + "="*80)
    print(f"SHEET: {sheet_name}")
    print("="*80)

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = (len(df) - non_null) / len(df) * 100
        dtype = str(df[col].dtype)
        print(f"  • {col:<20} ({dtype:<15}) - {non_null:>4}/{len(df)} filled ({null_pct:>5.1f}% missing)")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nLast 5 rows:")
    print(df.tail())

    # Check for numeric columns with actual data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric columns with data:")
    for col in numeric_cols:
        if df[col].notna().any():
            print(f"  • {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

# Try to understand the data structure
print("\n" + "="*80)
print("ATTEMPTING TO BUILD REVENUE DATA")
print("="*80)

# Load orders sheet
orders = pd.read_excel(file_path, sheet_name='orders')
products = pd.read_excel(file_path, sheet_name='products')
customers = pd.read_excel(file_path, sheet_name='customers')

print(f"\nOrders shape: {orders.shape}")
print(f"Products shape: {products.shape}")
print(f"Customers shape: {customers.shape}")

# Check if we can join to get prices
print(f"\nOrders columns: {orders.columns.tolist()}")
print(f"Products columns: {products.columns.tolist()}")
print(f"Customers columns: {customers.columns.tolist()}")

# Look for price information in products sheet
if 'Unit Price' in products.columns or 'Price' in products.columns:
    print("\n✓ Found price info in products sheet!")
    print("\nProducts sheet sample:")
    print(products.head(10))

    # Try to join
    if 'Product ID' in orders.columns and 'Product ID' in products.columns:
        print("\n✓ Can join on Product ID!")
        merged = orders.merge(products, on='Product ID', how='left', suffixes=('', '_product'))
        print(f"\nMerged shape: {merged.shape}")
        print(f"\nMerged columns: {merged.columns.tolist()}")

        # Check if we now have price
        price_col = None
        for col in merged.columns:
            if 'price' in col.lower() and merged[col].notna().any():
                price_col = col
                print(f"\n✓ Found price column after merge: '{price_col}'")
                print(f"  Non-null values: {merged[price_col].notna().sum()}/{len(merged)}")
                print(f"  Price range: ${merged[price_col].min():.2f} - ${merged[price_col].max():.2f}")
                break

        if price_col and 'Quantity' in merged.columns:
            print("\n✓ Can calculate revenue = Quantity × Price")
            merged['revenue'] = merged['Quantity'] * merged[price_col]
            print(f"  Revenue calculated: {merged['revenue'].notna().sum()} rows")
            print(f"  Total revenue: ${merged['revenue'].sum():,.2f}")
            print(f"  Mean transaction: ${merged['revenue'].mean():.2f}")

            # Aggregate to daily
            if 'Order Date' in merged.columns:
                merged['Order Date'] = pd.to_datetime(merged['Order Date'])
                daily = merged.groupby('Order Date')['revenue'].sum().reset_index()
                daily.columns = ['date', 'revenue']
                daily = daily.sort_values('date')

                print(f"\n✓ Daily aggregation successful!")
                print(f"  Days: {len(daily)}")
                print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
                print(f"  Mean daily: ${daily['revenue'].mean():.2f}")
                print(f"  Std daily: ${daily['revenue'].std():.2f}")

                # Save it
                daily.to_csv('/home/user/Coffee-shop/data/processed/raw_data_daily_revenue.csv', index=False)
                print(f"\n✓ Saved to: data/processed/raw_data_daily_revenue.csv")

                print(f"\nFirst 10 days:")
                print(daily.head(10))
                print(f"\nLast 10 days:")
                print(daily.tail(10))
else:
    print("\n✗ No price information found in products sheet")
    print("Cannot calculate revenue")
