# Coffee Shop - Feature Implementation Matrix

## Complete Mapping: Database Tables → Models → Controllers → Views

### CUSTOMERS & AUTHENTICATION

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **User Auth** | users, otp_codes | user.py | auth_controller.py | login_ex.py, register_ex.py | ✅ Complete (OTP placeholder) |
| **User Profiles** | users, user_preferences | user.py | user_controller.py | profile_ex.py | ✅ Complete |
| **Loyalty Points** | users, loyalty_points_history | user.py | user_controller.py | profile_ex.py | ✅ Complete |
| **Email/Phone Verify** | users (flags) | user.py | auth_controller.py | - | ⚠️ Partial (flags only) |

---

### PRODUCT CATALOG

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Categories** | categories | product.py | menu_controller.py, admin_category_controller.py | menu_ex.py, admin_categories_ex.py | ✅ Complete |
| **Products** | products | product.py | menu_controller.py, admin_product_controller.py | menu_ex.py, admin_products_ex.py | ✅ Complete |
| **Product Sizing** | product_sizes | product.py | menu_controller.py | menu_ex.py | ✅ Complete |
| **Toppings** | toppings | topping.py | menu_controller.py | menu_ex.py | ✅ Complete |
| **Reviews** | reviews, products | review.py | menu_controller.py | menu_ex.py | ✅ Complete |
| **Product Rating** | products, reviews | product.py | - | menu_ex.py | ✅ Complete (auto-calc) |
| **Featured Products** | products | product.py | menu_controller.py | menu_ex.py | ✅ Complete |

---

### SHOPPING & CART

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Shopping Cart** | cart | cart.py | cart_controller.py | cart_ex.py | ⚠️ Has BUG (p.image) |
| **Favorites** | favorites | N/A | favorites_controller.py | favorites_ex.py, menu_ex.py | ✅ Complete |
| **Cart Quantity** | cart | cart.py | cart_controller.py | cart_ex.py | ✅ Complete |
| **Customization** | cart | cart.py | cart_controller.py | cart_ex.py | ✅ Complete (sugar, ice, temp) |

---

### ORDERS & CHECKOUT

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Create Orders** | orders, order_items | order.py | order_controller.py | checkout_dialog.py | ✅ Complete |
| **Order Status** | orders | order.py | order_controller.py, admin_order_controller.py | orders_ex.py, admin_orders_ex.py | ✅ Complete |
| **Order History** | orders, order_items | order.py | order_controller.py | orders_ex.py | ✅ Complete |
| **Status History** | order_status_history | N/A | admin_order_controller.py | - | ⚠️ Minimal (table exists) |
| **Order Types** | orders | order.py | order_controller.py | checkout_dialog.py | ✅ Complete (pickup/delivery/dine-in) |

---

### VOUCHERS & DISCOUNTS

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Vouchers** | vouchers, user_vouchers | voucher.py | admin_voucher_controller.py | admin_vouchers_ex.py | ✅ Complete |
| **Discount Calc** | vouchers, orders | voucher.py | order_controller.py | checkout_dialog.py | ✅ Complete |
| **Voucher History** | voucher_usage | N/A | order_controller.py | - | ✅ Complete |
| **Usage Limits** | vouchers, user_vouchers | voucher.py | order_controller.py | - | ✅ Complete |

---

### LOYALTY & REWARDS

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Points Earning** | loyalty_points_history, users | user.py | order_controller.py | profile_ex.py | ✅ Complete |
| **Points Redemption** | loyalty_points_history, users | user.py | user_controller.py | profile_ex.py | ✅ Complete |
| **Membership Tiers** | users | user.py | user_controller.py | profile_ex.py | ✅ Complete |
| **Badges** | badges, user_badges | ❌ NONE | ❌ NONE | ❌ NONE | ❌ 0% |
| **Missions** | loyalty_missions, user_mission_progress | ❌ NONE | ❌ NONE | ❌ NONE | ❌ 0% |

---

### STORES & LOCATIONS

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Store Info** | stores | store.py | order_controller.py | checkout_dialog.py | ✅ Complete |
| **Proximity Search** | stores | store.py | - | - | ✅ Complete (distance calc) |

---

### NOTIFICATIONS

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Order Notifications** | notifications | notification.py | order_controller.py | main_window_ex.py | ✅ Complete (auto-created) |
| **Other Notifications** | notifications | notification.py | N/A | main_window_ex.py | ⚠️ Schema supports but minimal use |

---

### ADMIN FUNCTIONALITY

| Feature | DB Tables | Models | Controllers | Views | Status |
|---------|-----------|--------|-------------|-------|--------|
| **Admin Auth** | admin_users | N/A | admin_controller.py | admin_login_ex.py | ✅ Complete |
| **Dashboard** | orders, users, products | N/A | admin_controller.py | admin_dashboard_ex.py | ✅ Complete |
| **Product Mgmt** | products, categories | product.py | admin_product_controller.py | admin_products_ex.py | ✅ Complete |
| **Category Mgmt** | categories | product.py | admin_category_controller.py | admin_categories_ex.py | ✅ Complete |
| **Order Mgmt** | orders, order_items | order.py | admin_order_controller.py | admin_orders_ex.py | ✅ Complete |
| **User Mgmt** | users | user.py | admin_user_controller.py | admin_users_ex.py | ✅ Complete |
| **Voucher Mgmt** | vouchers | voucher.py | admin_voucher_controller.py | admin_vouchers_ex.py | ✅ Complete |
| **Activity Log** | admin_activity_log | N/A | admin_*.py | ❌ No view | ✅ Complete (backend) |
| **ML Analytics** | orders, stores | N/A | - | admin_ml_analytics_ex.py | ✅ Complete (revenue forecasting) |

---

## UNIMPLEMENTED FEATURES (Schema Exists, No Code)

### 1. Badge System (2 tables, 0% implementation)
```
Tables:    badges, user_badges
Models:    ❌ NONE
Controllers: ❌ NONE
Views:     ❌ NONE
Fields:    badges.icon_url (VARCHAR 500) - UNUSED
Status:    Complete schema definition, zero implementation
```

### 2. Loyalty Missions (2 tables, 0% implementation)
```
Tables:    loyalty_missions, user_mission_progress
Models:    ❌ NONE
Controllers: ❌ NONE
Views:     ❌ NONE
Status:    Complete schema definition, zero implementation
```

### 3. OTP Verification (1 table, stub implementation)
```
Tables:    otp_codes
Models:    ❌ NONE (logic in auth_controller)
Controllers: ⚠️ STUB - only prints to console
Views:     ❌ NONE
Issue:     Print statement: print(f"OTP Code: {otp_code} for {identifier}")
Status:    Placeholder, not functional
```

---

## BUGS & ISSUES

### Critical Bug #1
**File**: `models/cart.py` Line 58
```sql
SELECT c.*, p.image as product_image  # WRONG!
FROM cart c
```
**Correct**: Should be `p.image_url`
**Impact**: Cart operations will fail
**Branch**: `claude/refactor-db-schema-images-01KcVVKCBuX6H6XuRk8M8fhA`

---

## TABLE USAGE SUMMARY

### Fully Used (16 tables)
users, categories, products, toppings, product_sizes, reviews, cart, orders, order_items, vouchers, user_vouchers, voucher_usage, loyalty_points_history, stores, notifications, admin_users, admin_activity_log, favorites

### Partially Used (3 tables)
user_preferences (created but not auto-filled), saved_payment_methods (schema only), order_status_history (rarely updated)

### Unused (6 tables)
user_favorites (deprecated by favorites), otp_codes (stub), badges (schema only), user_badges (schema only), loyalty_missions (schema only), user_mission_progress (schema only)

---

## Code Paths Reference

```
Database Files:
  ├── database/schema.sql              (Main schema)
  ├── database/admin_schema.sql        (Admin tables)
  └── database/schema_updates.sql      (Updates & additions)

Models (Business Logic):
  ├── models/user.py                  (User, auth, loyalty points)
  ├── models/product.py               (Categories, products)
  ├── models/cart.py                  (Shopping cart) ⚠️ HAS BUG
  ├── models/order.py                 (Orders)
  ├── models/review.py                (Reviews)
  ├── models/topping.py               (Toppings)
  ├── models/voucher.py               (Vouchers)
  ├── models/notification.py          (Notifications)
  └── models/store.py                 (Stores)

Controllers (Business Operations):
  ├── controllers/auth_controller.py           (Login, OTP placeholder)
  ├── controllers/user_controller.py           (User profile)
  ├── controllers/menu_controller.py           (Products)
  ├── controllers/cart_controller.py           (Shopping)
  ├── controllers/favorites_controller.py      (Favorites)
  ├── controllers/order_controller.py          (Orders)
  ├── controllers/admin_controller.py          (Admin auth, dashboard)
  ├── controllers/admin_category_controller.py (Categories)
  ├── controllers/admin_product_controller.py  (Products)
  ├── controllers/admin_user_controller.py     (Users)
  ├── controllers/admin_order_controller.py    (Orders)
  └── controllers/admin_voucher_controller.py  (Vouchers)

Views (UI):
  ├── views/menu_ex.py                    (Product catalog)
  ├── views/cart_ex.py                    (Shopping cart)
  ├── views/orders_ex.py                  (Order history)
  ├── views/profile_ex.py                 (User profile)
  ├── views/favorites_ex.py               (Favorites)
  ├── views/checkout_dialog.py            (Checkout)
  ├── views/admin_*.py                    (Admin pages)
  └── main.py / admin.py                  (App entry points)

Utilities:
  └── utils/database.py                  (Database connection & queries)
```

