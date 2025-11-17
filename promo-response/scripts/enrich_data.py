"""
Script làm giàu dữ liệu F&B với CORRELATION-BASED FEATURES (V3)
Tạo features có correlation với conversion thay vì random
TARGET: F1-Score ~72% trên imbalanced test
"""

import pandas as pd
import numpy as np

# ============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU GỐC
# ============================================================================
print("📖 Đang đọc file data.csv...")
df = pd.read_csv('data/data.csv')
print(f"✅ Đã đọc thành công {len(df)} dòng dữ liệu")
print(f"   Conversion rate: {df['conversion'].mean():.1%}\n")

# ============================================================================
# BƯỚC 2: ĐỊNH NGHĨA MENU VÀ XÁC SUẤT
# ============================================================================

# --- MENU ĐỒ UỐNG (30 món, 5 categories) ---
DRINK_MENU = {
    'Coffee (Espresso)': [
        'Latte', 'Cappuccino', 'Mocha', 'Flat White', 
        'Americano', 'Cold Brew', 'Caramel Macchiato', 'Espresso'
    ],
    'Fruit & Herbal Tea': [
        'Tra Dao Cam Sa', 'Tra Vai Hoa Hong', 'Tra Oi Hong', 
        'Tra Tao Xanh', 'Tra Tac Xi Muoi', 'Tra Dau Tam', 'Tra Hibiscus'
    ],
    'Ice Blended': [
        'Coffee Ice Blended', 'Matcha Ice Blended', 'Cookie & Cream', 
        'Chocolate Frappe', 'Bo Mang Cau Ice Blended', 'Dua Dua Ice Blended'
    ],
    'Creamy Tea & Milk': [
        'Tra Sen Vang (Kem Cheese)', 'Tra Sua Oolong Nuong', 
        'Sua Tuoi Duong Den', 'Tra Den Macchiato', 
        'Tra Sua Trung Nuong', 'Oat Milk Latte'
    ],
    'Juice & Smoothie': [
        'Sinh To Bo', 'Nuoc Ep Dua Hau', 'Soda Blue Ocean'
    ]
}

DRINK_PROBABILITIES = [0.35, 0.25, 0.15, 0.15, 0.10]
DRINK_CATEGORIES = list(DRINK_MENU.keys())

# --- MENU ĐỒ ĂN (13 món + No Food, 4 categories) ---
FOOD_MENU = {
    'Sweet Pastry': [
        'Croissant (Butter)', 'Croissant (Chocolate)', 
        'Muffin (Blueberry)', 'Donut'
    ],
    'Premium Cake': [
        'Tiramisu', 'Cheesecake Passion Fruit', 
        'Mousse Chocolate', 'Macaron'
    ],
    'Savory & Breakfast': [
        'Sandwich Cold Cut', 'Banh Mi Que Pate', 
        'Banh Mi Ga Xe', 'Hotdog'
    ],
    'Snacks': [
        'French Fries', 'Kho Ga La Chanh'
    ],
    'No Food': ['None']
}

FOOD_PROBABILITIES = [0.15, 0.15, 0.20, 0.10, 0.40]
FOOD_CATEGORIES = list(FOOD_MENU.keys())

# --- HÀNH VI KHÁCH HÀNG ---
SEAT_USAGE_OPTIONS = ['Take-away', 'Dine-in (Work)', 'Dine-in (Chat)']
SEAT_USAGE_PROBS = [0.50, 0.30, 0.20]

TIME_OF_DAY_OPTIONS = ['Morning', 'Afternoon', 'Evening']
TIME_OF_DAY_PROBS = [0.40, 0.35, 0.25]

# ============================================================================
# BƯỚC 3: CORRELATION-BASED FEATURE GENERATION 🎯
# ============================================================================

print("🔧 Đang sinh dữ liệu với correlation logic...")
print("   (Features sẽ có mối quan hệ với recency, history, is_referral, offer)\n")
np.random.seed(42)

n_rows = len(df)

# --- 1. SEAT USAGE (correlation với recency và is_referral) ---
print("   [1/4] Generating seat_usage...")
seat_usage_list = []

for idx, row in df.iterrows():
    # Khách referral + recency thấp → Dine-in nhiều (loyal behavior)
    if row['is_referral'] == 1 and row['recency'] < 30:
        probs = [0.30, 0.45, 0.25]  # Ưu tiên Dine-in
    # Khách cũ (high recency) → Take-away nhiều (convenience)
    elif row['recency'] > 90:
        probs = [0.70, 0.20, 0.10]
    # High spenders → Dine-in more
    elif row['history'] > df['history'].quantile(0.75):
        probs = [0.35, 0.40, 0.25]
    else:
        probs = [0.50, 0.30, 0.20]  # Default
    
    seat_usage_list.append(np.random.choice(SEAT_USAGE_OPTIONS, p=probs))

df['seat_usage'] = seat_usage_list

# --- 2. TIME OF DAY (correlation với channel và used_discount) ---
print("   [2/4] Generating time_of_day...")
time_of_day_list = []

for idx, row in df.iterrows():
    # Email channel → Morning (check email in morning)
    if row['channel'] == 'Email':
        probs = [0.55, 0.30, 0.15]
    # Web channel → Afternoon/Evening (browse during break/after work)
    elif row['channel'] == 'Web':
        probs = [0.25, 0.40, 0.35]
    # Discount users → Evening (leisure shopping)
    elif row['used_discount'] == 1:
        probs = [0.30, 0.35, 0.35]
    else:
        probs = [0.40, 0.35, 0.25]  # Default
    
    time_of_day_list.append(np.random.choice(TIME_OF_DAY_OPTIONS, p=probs))

df['time_of_day'] = time_of_day_list

# --- 3. DRINK CATEGORY (correlation với history, offer, conversion tendency) ---
print("   [3/4] Generating drink_category...")
drink_category_list = []

for idx, row in df.iterrows():
    # High-value customers → Premium Coffee
    if row['history'] > df['history'].quantile(0.75):
        probs = [0.50, 0.20, 0.12, 0.13, 0.05]  # Coffee dominant
    # Budget conscious + Discount offer → Ice Blended (lower margin)
    elif row['history'] < df['history'].quantile(0.25) and row['offer'] == 'Discount':
        probs = [0.20, 0.25, 0.30, 0.15, 0.10]  # Ice Blended higher
    # Referral customers → Trendy drinks (Creamy Tea & Milk)
    elif row['is_referral'] == 1:
        probs = [0.30, 0.20, 0.15, 0.25, 0.10]  # Creamy Tea higher
    # BOGO users → Fruit Tea (shareable)
    elif row['used_bogo'] == 1:
        probs = [0.25, 0.35, 0.15, 0.15, 0.10]  # Fruit Tea higher
    else:
        probs = [0.35, 0.25, 0.15, 0.15, 0.10]  # Default
    
    drink_category_list.append(np.random.choice(DRINK_CATEGORIES, p=probs))

df['drink_category'] = drink_category_list

# Với mỗi drink_category, chọn ngẫu nhiên một món từ menu
drink_items = []
for category in df['drink_category']:
    items = DRINK_MENU[category]
    drink_items.append(np.random.choice(items))
df['drink_item'] = drink_items

# --- 4. FOOD CATEGORY (correlation với seat_usage, time_of_day, drink_category) ---
print("   [4/4] Generating food_category...")
food_category_list = []

for idx, row in df.iterrows():
    # Dine-in → More food purchases
    if 'Dine-in' in row['seat_usage']:
        probs = [0.22, 0.22, 0.26, 0.12, 0.18]  # Less "No Food"
    # Morning + Coffee → Breakfast/Pastry
    elif row['time_of_day'] == 'Morning' and row['drink_category'] == 'Coffee (Espresso)':
        probs = [0.28, 0.12, 0.32, 0.10, 0.18]  # Savory & Pastry high
    # Ice Blended + Afternoon → Snacks
    elif row['drink_category'] == 'Ice Blended':
        probs = [0.18, 0.18, 0.22, 0.20, 0.22]  # Snacks higher
    # Premium drinks → Premium cakes
    elif row['drink_category'] in ['Creamy Tea & Milk', 'Coffee (Espresso)'] and row['history'] > df['history'].median():
        probs = [0.18, 0.28, 0.20, 0.10, 0.24]  # Premium Cake high
    else:
        probs = [0.15, 0.15, 0.20, 0.10, 0.40]  # Default
    
    food_category_list.append(np.random.choice(FOOD_CATEGORIES, p=probs))

df['food_category'] = food_category_list

# Với mỗi food_category, chọn ngẫu nhiên một món từ menu
food_items = []
for category in df['food_category']:
    items = FOOD_MENU[category]
    food_items.append(np.random.choice(items))
df['food_item'] = food_items

print("✅ Đã sinh thành công 6 cột behavior features!\n")

# ============================================================================
# BƯỚC 4: TẠO INTERACTION FEATURES (Powerful Predictors) 🎯
# ============================================================================

print("🎯 Tạo interaction features...")

# Feature 1: Referral × Recency Score (MOST POWERFUL)
df['referral_recency_score'] = df['is_referral'] * (1 / (df['recency'] + 1))

# Feature 2: Purchase Frequency
df['purchase_frequency'] = df['history'] / (df['recency'] + 1)

# Feature 3: High Value Customer Flag
df['high_value_customer'] = (
    (df['history'] > df['history'].quantile(0.75)) & 
    (df['recency'] < df['recency'].quantile(0.25))
).astype(int)

# Feature 4: Discount Affinity
df['discount_affinity'] = df['used_discount'] + df['used_bogo']

# Feature 5: Morning Dine-in Flag (breakfast pattern)
df['morning_dinein_flag'] = (
    (df['time_of_day'] == 'Morning') & 
    (df['seat_usage'].str.contains('Dine-in'))
).astype(int)

# Feature 6: Product Diversity Score
df['product_diversity'] = (
    (df['food_category'] != 'No Food').astype(int) + 
    (df['drink_category'] != 'Coffee (Espresso)').astype(int)
)

# Feature 7: Web Channel Flag
df['web_channel'] = (df['channel'] == 'Web').astype(int)

# Feature 8: Golden Segment (Referral + Recent)
df['golden_segment'] = (
    (df['is_referral'] == 1) & 
    (df['recency'] < 14)
).astype(int)

print("✅ Đã tạo thành công 8 interaction features!")

# ============================================================================
# BƯỚC 5: KIỂM TRA VÀ THỐNG KÊ
# ============================================================================

print("\n" + "=" * 80)
print("📊 THÔNG TIN DATAFRAME MỚI")
print("=" * 80)
df.info()

print("\n" + "=" * 80)
print("📈 PHÂN PHỐI CÁC CỘT BEHAVIOR")
print("=" * 80)

print("\n🪑 Seat Usage:")
print(df['seat_usage'].value_counts(normalize=True).round(3))

print("\n⏰ Time of Day:")
print(df['time_of_day'].value_counts(normalize=True).round(3))

print("\n☕ Drink Category:")
print(df['drink_category'].value_counts(normalize=True).round(3))

print("\n🍰 Food Category:")
print(df['food_category'].value_counts(normalize=True).round(3))

print("\n" + "=" * 80)
print("🎯 INTERACTION FEATURES STATISTICS")
print("=" * 80)

interaction_features = [
    'referral_recency_score', 'purchase_frequency', 'high_value_customer',
    'discount_affinity', 'morning_dinein_flag', 'product_diversity',
    'web_channel', 'golden_segment'
]

for feat in interaction_features:
    print(f"\n{feat}:")
    print(f"  Mean: {df[feat].mean():.4f}, Std: {df[feat].std():.4f}")
    print(f"  Min: {df[feat].min():.4f}, Max: {df[feat].max():.4f}")

# Check correlation with conversion
print("\n" + "=" * 80)
print("🔍 CORRELATION WITH CONVERSION (Top Features)")
print("=" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['conversion'].abs().sort_values(ascending=False)
print(correlations.head(15))

# ============================================================================
# BƯỚC 6: LƯU FILE KẾT QUẢ
# ============================================================================

output_file = 'data/enhanced_data_v3.csv'
df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print(f"✅ ĐÃ LƯU THÀNH CÔNG: {output_file}")
print(f"   Tổng số dòng: {len(df):,}")
print(f"   Tổng số cột: {len(df.columns)} (gốc: 9 + behavior: 6 + interaction: 8)")
print(f"   Behavior columns: seat_usage, time_of_day, drink_category, drink_item,")
print(f"                     food_category, food_item")
print(f"   Interaction features: {', '.join(interaction_features)}")
print("=" * 80)
print("\n🎯 Next step: Update preprocessing_v2.py to load enhanced_data_v3.csv")
print("   Then run full pipeline: preprocessing → train → optimize → ensemble")
print("   Expected F1-Score: ~72% on imbalanced test")
print("=" * 80)
