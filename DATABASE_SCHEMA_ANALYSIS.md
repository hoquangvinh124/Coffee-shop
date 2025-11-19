# Coffee Shop Database Schema Analysis

## COMPREHENSIVE DATABASE SCHEMA OVERVIEW

### Database Summary
- **Database Name**: coffee_shop
- **Total Tables**: 29 (excluding admin tables: 25 customer-facing tables)
- **Engine**: InnoDB with UTF8MB4 encoding

---

## PART 1: ALL DATABASE TABLES AND COLUMNS

### Core User Tables

#### 1. **users** (Core user data)
```
id (INT) - Primary Key, Auto Increment
email (VARCHAR 255) - UNIQUE, Indexed
phone (VARCHAR 20) - UNIQUE, Indexed
password_hash (VARCHAR 255)
full_name (VARCHAR 255)
date_of_birth (DATE)
avatar_url (VARCHAR 500)
membership_tier (ENUM: 'Bronze', 'Silver', 'Gold') - Default: Bronze, Indexed
loyalty_points (INT) - Default: 0
created_at (TIMESTAMP)
updated_at (TIMESTAMP)
last_login (TIMESTAMP)
is_active (BOOLEAN) - Default: TRUE
email_verified (BOOLEAN) - Default: FALSE
phone_verified (BOOLEAN) - Default: FALSE
```

#### 2. **user_preferences** (User customization preferences)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
favorite_size (ENUM: 'S', 'M', 'L') - Default: M
favorite_sugar_level (INT) - Default: 50
favorite_ice_level (INT) - Default: 50
preferred_toppings (JSON)
allergies (JSON)
```

#### 3. **otp_codes** (One-Time Password verification)
⚠️ **TABLE EXISTS BUT MOSTLY UNUSED/PLACEHOLDER**
```
id (INT) - Primary Key
user_id (INT)
email (VARCHAR 255) - Indexed with otp_code
phone (VARCHAR 20) - Indexed with otp_code
otp_code (VARCHAR 6) - NOT NULL
purpose (ENUM: 'registration', 'login', 'password_reset')
expires_at (TIMESTAMP)
is_used (BOOLEAN) - Default: FALSE
created_at (TIMESTAMP)
```
**Status**: Table exists; auth_controller has OTP code generation but only prints to console - NO actual email/SMS implementation

---

### Product & Category Tables

#### 4. **categories** (Product categories)
```
id (INT) - Primary Key
name (VARCHAR 100) - NOT NULL
name_en (VARCHAR 100) - English name
description (TEXT)
icon_url (VARCHAR 500) - URL to icon image
icon (VARCHAR 10) - Emoji icon (Added in schema_updates.sql) - Default: '☕'
display_order (INT) - Default: 0
is_active (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
updated_at (TIMESTAMP) - Added in schema_updates.sql
```

#### 5. **products** (Coffee and food items)
```
id (INT) - Primary Key
category_id (INT) - Foreign Key → categories.id (ON DELETE CASCADE) - Indexed
name (VARCHAR 255) - NOT NULL
name_en (VARCHAR 255)
description (TEXT)
ingredients (TEXT)
allergens (JSON)
image_url (VARCHAR 500) - Product image
base_price (DECIMAL 10,2)
calories_small (INT)
calories_medium (INT)
calories_large (INT)
is_hot (BOOLEAN) - Default: TRUE
is_cold (BOOLEAN) - Default: TRUE
is_caffeine_free (BOOLEAN) - Default: FALSE
is_available (BOOLEAN) - Default: TRUE - Indexed
is_featured (BOOLEAN) - Default: FALSE - Indexed
is_new (BOOLEAN) - Default: FALSE (Added in schema_updates.sql)
is_bestseller (BOOLEAN) - Default: FALSE (Added in schema_updates.sql)
is_seasonal (BOOLEAN) - Default: FALSE (Added in schema_updates.sql)
rating (DECIMAL 3,2) - Default: 0
total_reviews (INT) - Default: 0
created_at (TIMESTAMP)
updated_at (TIMESTAMP)
```

#### 6. **toppings** (Add-on toppings)
```
id (INT) - Primary Key
name (VARCHAR 100) - NOT NULL
name_en (VARCHAR 100)
price (DECIMAL 10,2)
calories (INT) - Default: 0
is_available (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
```

#### 7. **product_sizes** (Size pricing for products)
```
id (INT) - Primary Key
product_id (INT) - Foreign Key → products.id (ON DELETE CASCADE)
size (ENUM: 'S', 'M', 'L') - NOT NULL
price_adjustment (DECIMAL 10,2) - Default: 0
UNIQUE: (product_id, size)
```

#### 8. **reviews** (Product reviews)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
product_id (INT) - Foreign Key → products.id (ON DELETE CASCADE) - Indexed
order_id (INT) - Foreign Key → orders.id (ON DELETE CASCADE)
rating (INT) - CHECK (1-5) - NOT NULL - Indexed
comment (TEXT)
service_rating (INT) - CHECK (1-5)
images (JSON) - Array of review images
is_approved (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
```

---

### Shopping & Cart Tables

#### 9. **cart** (Active shopping cart)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE) - NOT NULL
product_id (INT) - Foreign Key → products.id (ON DELETE CASCADE) - NOT NULL
size (ENUM: 'S', 'M', 'L') - NOT NULL
quantity (INT) - NOT NULL
sugar_level (INT) - Default: 50
ice_level (INT) - Default: 50
temperature (ENUM: 'hot', 'cold') - Default: 'cold'
toppings (JSON) - Array of topping IDs
created_at (TIMESTAMP)
updated_at (TIMESTAMP)
```

#### 10. **user_favorites** / **favorites** (User favorite products)
⚠️ **NOTE: Both tables exist! Possible duplicate/migration issue**
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE) - NOT NULL
product_id (INT) - Foreign Key → products.id (ON DELETE CASCADE) - NOT NULL
created_at (TIMESTAMP)
UNIQUE: (user_id, product_id)
Indexes: idx_user, idx_product
```
**Status**: Actively used via `favorites` table in schema_updates.sql

---

### Order Tables

#### 11. **orders** (Customer orders)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id - Indexed
order_number (VARCHAR 50) - UNIQUE
store_id (INT) - Foreign Key → stores.id
order_type (ENUM: 'pickup', 'delivery', 'dine_in')
status (ENUM: 'pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled') - Default: 'pending' - Indexed
subtotal (DECIMAL 10,2)
discount_amount (DECIMAL 10,2) - Default: 0
delivery_fee (DECIMAL 10,2) - Default: 0
total_amount (DECIMAL 10,2)
payment_method (ENUM: 'cash', 'momo', 'shopeepay', 'zalopay', 'applepay', 'googlepay', 'card')
payment_status (ENUM: 'pending', 'paid', 'failed', 'refunded') - Default: 'pending'
delivery_address (TEXT)
delivery_latitude (DECIMAL 10,8)
delivery_longitude (DECIMAL 11,8)
table_number (VARCHAR 20)
qr_code (VARCHAR 100)
notes (TEXT)
estimated_ready_time (TIMESTAMP)
completed_at (TIMESTAMP)
cancelled_at (TIMESTAMP)
cancellation_reason (TEXT)
created_at (TIMESTAMP)
updated_at (TIMESTAMP)
```

#### 12. **order_items** (Individual items in orders)
```
id (INT) - Primary Key
order_id (INT) - Foreign Key → orders.id (ON DELETE CASCADE)
product_id (INT) - Foreign Key → products.id
product_name (VARCHAR 255) - NOT NULL
size (ENUM: 'S', 'M', 'L')
quantity (INT)
unit_price (DECIMAL 10,2)
sugar_level (INT) - Default: 50
ice_level (INT) - Default: 50
temperature (ENUM: 'hot', 'cold') - Default: 'cold'
toppings (JSON)
topping_cost (DECIMAL 10,2) - Default: 0
subtotal (DECIMAL 10,2)
notes (TEXT)
```

#### 13. **order_status_history** (Track order status changes)
```
id (INT) - Primary Key
order_id (INT) - Foreign Key → orders.id (ON DELETE CASCADE) - Indexed
old_status (ENUM: 'pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled')
new_status (ENUM: 'pending', 'confirmed', 'preparing', 'ready', 'delivering', 'completed', 'cancelled') - NOT NULL
changed_by_admin_id (INT) - Foreign Key → admin_users.id (ON DELETE SET NULL)
notes (TEXT)
created_at (TIMESTAMP) - Indexed
```

---

### Voucher & Discount Tables

#### 14. **vouchers** (Discount vouchers/coupons)
```
id (INT) - Primary Key
code (VARCHAR 50) - UNIQUE - Indexed
name (VARCHAR 255)
description (TEXT)
discount_type (ENUM: 'percentage', 'fixed')
discount_value (DECIMAL 10,2)
min_order_amount (DECIMAL 10,2) - Default: 0
max_discount_amount (DECIMAL 10,2)
usage_limit (INT) - Total times voucher can be used
usage_per_user (INT) - Default: 1
current_usage (INT) - Default: 0
start_date (TIMESTAMP)
end_date (TIMESTAMP)
is_active (BOOLEAN) - Default: TRUE - Indexed
applicable_to (ENUM: 'all', 'specific_products', 'specific_categories') - Default: 'all'
created_at (TIMESTAMP)
```

#### 15. **user_vouchers** (Track user voucher usage)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
voucher_id (INT) - Foreign Key → vouchers.id (ON DELETE CASCADE)
times_used (INT) - Default: 0
last_used_at (TIMESTAMP)
UNIQUE: (user_id, voucher_id)
```

#### 16. **voucher_usage** (Detailed voucher usage history)
```
id (INT) - Primary Key
voucher_id (INT) - Foreign Key → vouchers.id (ON DELETE CASCADE) - Indexed
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE) - Indexed
order_id (INT) - Foreign Key → orders.id (ON DELETE SET NULL)
discount_amount (DECIMAL 10,2)
used_at (TIMESTAMP) - Indexed
```

---

### Loyalty & Rewards Tables

#### 17. **loyalty_missions** (Loyalty challenges/quests)
⚠️ **TABLE EXISTS BUT NO MODEL/CONTROLLER IMPLEMENTATION**
```
id (INT) - Primary Key
name (VARCHAR 255)
description (TEXT)
mission_type (ENUM: 'order_count', 'product_specific', 'total_spent', 'birthday')
target_value (INT) - NOT NULL
reward_type (ENUM: 'points', 'voucher', 'badge', 'free_item')
reward_value (VARCHAR 255)
start_date (TIMESTAMP)
end_date (TIMESTAMP)
is_active (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
```

#### 18. **user_mission_progress** (Track user progress on missions)
⚠️ **TABLE EXISTS BUT NO MODEL/CONTROLLER IMPLEMENTATION**
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
mission_id (INT) - Foreign Key → loyalty_missions.id (ON DELETE CASCADE)
current_value (INT) - Default: 0
is_completed (BOOLEAN) - Default: FALSE
completed_at (TIMESTAMP)
UNIQUE: (user_id, mission_id)
```

#### 19. **loyalty_points_history** (Track point transactions)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE) - Indexed
points (INT)
transaction_type (ENUM: 'earn', 'redeem', 'expire', 'adjust')
description (VARCHAR 255)
order_id (INT) - Foreign Key → orders.id
created_at (TIMESTAMP)
```

#### 20. **badges** (Achievement badges)
⚠️ **TABLE EXISTS BUT NO MODEL/CONTROLLER IMPLEMENTATION**
```
id (INT) - Primary Key
name (VARCHAR 100)
description (TEXT)
icon_url (VARCHAR 500) - Badge icon/image
requirement (TEXT)
created_at (TIMESTAMP)
```

#### 21. **user_badges** (Track earned badges)
⚠️ **TABLE EXISTS BUT NO MODEL/CONTROLLER IMPLEMENTATION**
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
badge_id (INT) - Foreign Key → badges.id (ON DELETE CASCADE)
earned_at (TIMESTAMP)
UNIQUE: (user_id, badge_id)
```

---

### Store/Location Tables

#### 22. **stores** (Coffee shop branches)
```
id (INT) - Primary Key
name (VARCHAR 255) - NOT NULL
address (TEXT) - NOT NULL
city (VARCHAR 100)
district (VARCHAR 100)
latitude (DECIMAL 10,8)
longitude (DECIMAL 11,8)
phone (VARCHAR 20)
opening_time (TIME)
closing_time (TIME)
is_active (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
```

---

### Notification & Communication Tables

#### 23. **notifications** (User notifications)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
title (VARCHAR 255)
message (TEXT)
notification_type (ENUM: 'order_update', 'promotion', 'loyalty', 'system')
is_read (BOOLEAN) - Default: FALSE
related_order_id (INT)
created_at (TIMESTAMP)
Index: (user_id, is_read)
```

#### 24. **saved_payment_methods** (User payment methods)
```
id (INT) - Primary Key
user_id (INT) - Foreign Key → users.id (ON DELETE CASCADE)
payment_type (ENUM: 'momo', 'shopeepay', 'zalopay', 'card')
account_info (VARCHAR 255) - Masked account details
is_default (BOOLEAN) - Default: FALSE
created_at (TIMESTAMP)
```

---

### Admin Tables

#### 25. **admin_users** (Admin staff accounts)
```
id (INT) - Primary Key
username (VARCHAR 100) - UNIQUE - Indexed
email (VARCHAR 255) - UNIQUE - Indexed
password_hash (VARCHAR 255)
full_name (VARCHAR 255)
role (ENUM: 'super_admin', 'admin', 'manager', 'staff') - Default: 'staff'
is_active (BOOLEAN) - Default: TRUE
created_at (TIMESTAMP)
updated_at (TIMESTAMP)
last_login (TIMESTAMP)
```

#### 26. **admin_activity_log** (Audit trail for admin actions)
✅ **TABLE EXISTS AND IS BEING USED**
```
id (INT) - Primary Key
admin_id (INT) - Foreign Key → admin_users.id (ON DELETE CASCADE) - Indexed
action (VARCHAR 100) - Indexed
table_name (VARCHAR 100)
record_id (INT)
old_value (JSON) - Before update values
new_value (JSON) - After update values
ip_address (VARCHAR 45)
created_at (TIMESTAMP) - Indexed
```
**Status**: Actively used throughout admin controllers (AdminController.log_activity())

---

## PART 2: ICON AND IMAGE FIELD USAGE

### Icon Fields

| Table | Field | Type | Purpose | Status |
|-------|-------|------|---------|--------|
| **categories** | `icon_url` | VARCHAR(500) | URL to category icon image | Defined in schema.sql |
| **categories** | `icon` | VARCHAR(10) | Emoji icon (☕, 🥤, 🍰, 🧋) | Added in schema_updates.sql |
| **badges** | `icon_url` | VARCHAR(500) | Badge icon/image URL | No implementation |

### Image Fields

| Table | Field | Type | Purpose | Usage |
|-------|-------|------|---------|-------|
| **products** | `image_url` | VARCHAR(500) | Product photo | ✅ Used in queries |
| **users** | `avatar_url` | VARCHAR(500) | User profile picture | ✅ Used in user profile |
| **reviews** | `images` | JSON | Array of review photos | Defined but minimal use |

### BUG FOUND ⚠️
**File**: `/home/user/Coffee-shop/models/cart.py` (Line 58)
```python
SELECT c.*, p.name as product_name, p.name_en as product_name_en,
       p.image as product_image,  # ❌ WRONG - Field doesn't exist!
       p.base_price, p.is_available
FROM cart c
```
**Issue**: Trying to select `p.image` but the schema defines `products.image_url`
**Impact**: This query will fail when executed - likely a leftover from refactoring
**Related to current branch**: `claude/refactor-db-schema-images-01KcVVKCBuX6H6XuRk8M8fhA`

---

## PART 3: FEATURES IMPLEMENTATION STATUS

### ✅ FULLY IMPLEMENTED FEATURES

1. **User Authentication & Profiles**
   - Login/Registration (via auth_controller.py)
   - Email/Phone verification flags (partial)
   - User profile updates
   - Membership tiers (Bronze, Silver, Gold)
   - Avatar URLs support
   - Models: user.py

2. **Product Catalog**
   - Categories with names (Vietnamese + English)
   - Products with pricing, calories, description
   - Product filtering (hot, cold, caffeine-free, featured, new, bestseller, seasonal)
   - Product rating system (auto-calculated from reviews)
   - Models: product.py, category in product.py

3. **Shopping Cart**
   - Add/remove items
   - Quantity updates
   - Customization (sugar, ice, temperature, toppings)
   - Models: cart.py

4. **Orders & Order Management**
   - Create orders from cart
   - Order status tracking (pending → confirmed → preparing → ready → delivering → completed/cancelled)
   - Order types: pickup, delivery, dine-in
   - Payment methods: cash, momo, shopeepay, zalopay, applepay, googlepay, card
   - Payment status tracking
   - Order history
   - Models: order.py

5. **Reviews & Ratings**
   - Product reviews with ratings (1-5 stars)
   - Service ratings
   - Review images (JSON array)
   - Auto-update product ratings
   - Models: review.py

6. **Toppings/Add-ons**
   - Predefined toppings with prices
   - Per-item topping selection
   - Calorie calculation
   - Models: topping.py

7. **Vouchers & Discounts**
   - Fixed and percentage discounts
   - Usage limits (global and per-user)
   - Min order amount requirement
   - Date range validity
   - Voucher usage tracking
   - Models: voucher.py

8. **Loyalty System (Partial)**
   - Loyalty points earning (based on order total)
   - Points redemption
   - Points history tracking
   - Membership tier auto-upgrade based on points
   - Models: user.py (methods for points)

9. **Store/Branch Management**
   - Multiple store locations
   - Store details (address, hours, contact)
   - Proximity search (distance calculation)
   - Models: store.py

10. **Notifications**
    - Order status notifications
    - Promotion notifications
    - Loyalty notifications
    - Unread status tracking
    - Models: notification.py

11. **Favorites/Wishlist**
    - Add/remove products to favorites
    - Favorite products listing
    - Toggle favorite status
    - Controllers: favorites_controller.py
    - Tables: favorites (preferred) or user_favorites (deprecated?)

12. **Admin Functionality**
    - Admin authentication
    - Dashboard statistics
    - Order management
    - Product management (CRUD)
    - Category management
    - Voucher management
    - Activity logging (audit trail)
    - Controllers: admin_controller.py + specialized admin controllers

13. **Stored Payment Methods**
    - Save payment methods (optional)
    - Default payment method
    - Models: table exists, minimal implementation

---

### ⚠️ PARTIALLY IMPLEMENTED FEATURES

1. **User Preferences**
   - Table exists with favorite_size, sugar_level, ice_level
   - Preferred toppings and allergies stored as JSON
   - **But**: Minimal usage in UI/controllers - mostly stored but not actively used for auto-filling

2. **Email/Phone Verification**
   - Flags exist in users table (`email_verified`, `phone_verified`)
   - Methods exist in user.py (verify_email(), verify_phone())
   - **But**: No integration with actual email/SMS service

3. **Payment Processing**
   - Payment method enum and saved_payment_methods table
   - **But**: No actual payment gateway integration - likely placeholder

---

### ❌ PLACEHOLDER/UNUSED FEATURES (Database tables exist but no real implementation)

1. **OTP Codes (One-Time Password)**
   - ✗ Table: `otp_codes` exists with full schema
   - ✗ Code: `auth_controller.py` has OTP generation logic
   - ✗ **PROBLEM**: OTP is only printed to console: `print(f"OTP Code: {otp_code} for {identifier}")`
   - ✗ NO email/SMS sending implementation
   - ✗ NO actual verification flow in UI
   - Status: Database structure ready, but business logic is just a stub/placeholder

2. **Badges/Achievement System**
   - ✗ Table: `badges` exists (fields: name, description, icon_url, requirement)
   - ✗ Table: `user_badges` exists (fields: user_id, badge_id, earned_at)
   - ✗ **NO models, controllers, or views** for badges
   - ✗ Not integrated into user profile
   - Status: Schema exists only, 0% implementation

3. **Loyalty Missions/Challenges**
   - ✗ Table: `loyalty_missions` exists (fields: name, mission_type, target_value, reward_type, reward_value)
   - ✗ Table: `user_mission_progress` exists (fields: user_id, mission_id, current_value, is_completed)
   - ✗ **NO models, controllers, or views** for missions
   - ✗ Not exposed in UI
   - Status: Schema exists only, 0% implementation

4. **Order Status History Detailed Tracking**
   - ✗ Table: `order_status_history` exists
   - ✗ **NO controllers** actively using it for detailed tracking
   - ✗ Only status update exists, not history tracking
   - Status: Minimal use, mostly unused

---

### 🔍 DUPLICATE/CONFLICTING TABLES

1. **Favorites: user_favorites vs favorites**
   - **user_favorites**: Defined in schema.sql
   - **favorites**: Created in schema_updates.sql
   - **Current implementation**: Uses `favorites` table from schema_updates.sql
   - **Status**: user_favorites appears to be deprecated/replaced

---

## PART 4: IMAGE/ICON FIELD SUMMARY

### Current Icon Usage in Application

1. **Category Icons (Emoji)**
   - Implementation: `views/admin_categories_ex.py` allows editing emoji icons
   - Field: `categories.icon` (VARCHAR 10)
   - Values: ☕ (coffee), 🥤 (beverage), 🍰 (pastry), 🧋 (boba tea)
   - Status: ✅ Fully implemented

2. **Product Images**
   - Field: `products.image_url` (VARCHAR 500)
   - Usage: Menu display, order items, favorites, reviews
   - Status: ✅ Used throughout
   - **BUG**: cart.py references non-existent `p.image` field

3. **User Avatars**
   - Field: `users.avatar_url` (VARCHAR 500)
   - Usage: User profile display
   - Status: ✅ Supported but minimal usage

4. **Badge Icons (Unused)**
   - Field: `badges.icon_url` (VARCHAR 500)
   - Status: ⚠️ Table exists but badges feature not implemented

---

## PART 5: DATABASE SCHEMA STRUCTURE DIAGRAM

```
CUSTOMERS
├── users
│   ├── user_preferences
│   ├── otp_codes (PLACEHOLDER)
│   ├── user_badges (NOT IMPLEMENTED)
│   ├── loyalty_points_history
│   ├── user_favorites OR favorites
│   ├── user_vouchers
│   ├── saved_payment_methods
│   └── notifications
│
PRODUCTS & CATALOG
├── categories
│   ├── (icon_url, icon)
│   └── products
│       ├── reviews
│       ├── product_sizes
│       └── toppings
│
SHOPPING & ORDERS
├── cart
├── orders
│   ├── order_items
│   ├── order_status_history (MINIMAL USE)
│   └── reviews
│
VOUCHERS & LOYALTY
├── vouchers
│   └── voucher_usage, user_vouchers
├── loyalty_missions (NOT IMPLEMENTED)
│   └── user_mission_progress (NOT IMPLEMENTED)
└── badges (NOT IMPLEMENTED)
    └── user_badges (NOT IMPLEMENTED)
│
LOCATIONS
└── stores
│
ADMIN
├── admin_users
└── admin_activity_log (ACTIVELY USED)
```

---

## PART 6: KEY FINDINGS & RECOMMENDATIONS

### Critical Issues
1. **❌ cart.py uses `p.image` instead of `p.image_url`** - Will cause SQL errors
2. **⚠️ Multiple image column naming** (image, image_url, avatar_url) - inconsistent
3. **⚠️ OTP feature is placeholder only** - no actual SMS/email sending

### Unused Database Resources
- Badges system (3 tables): 0% implemented
- Loyalty missions (2 tables): 0% implemented  
- OTP codes: 0% functional (placeholder only)
- Order status history: Created but rarely updated

### Data Duplication
- `user_favorites` and `favorites` tables both exist for same purpose
- Code uses `favorites` (from schema_updates.sql), making `user_favorites` redundant

