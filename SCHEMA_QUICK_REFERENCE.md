# Coffee Shop Database Schema - Quick Reference

## Database Tables Overview

| Category | Table | Columns | Status | Implementation |
|----------|-------|---------|--------|-----------------|
| **Core Users** | users | 15 | ✅ Active | Full (auth, profile, loyalty) |
| | user_preferences | 5 | ✅ Active | Partial (stored but not used) |
| | otp_codes | 8 | ⚠️ Placeholder | Schema only - prints to console |
| **Products** | categories | 8 | ✅ Active | Full (icon_url + icon emoji) |
| | products | 20 | ✅ Active | Full (note: BUG in cart.py!) |
| | toppings | 6 | ✅ Active | Full |
| | product_sizes | 4 | ✅ Active | Full |
| | reviews | 9 | ✅ Active | Full |
| **Shopping** | cart | 9 | ✅ Active | Full (BUG: uses p.image instead of p.image_url) |
| | user_favorites | 3 | ⚠️ Deprecated | Replaced by "favorites" table |
| | favorites | 3 | ✅ Active | Full (in schema_updates.sql) |
| **Orders** | orders | 25 | ✅ Active | Full |
| | order_items | 14 | ✅ Active | Full |
| | order_status_history | 5 | ⚠️ Minimal Use | Schema exists but rarely used |
| **Vouchers** | vouchers | 13 | ✅ Active | Full |
| | user_vouchers | 4 | ✅ Active | Full |
| | voucher_usage | 5 | ✅ Active | Full |
| **Loyalty** | loyalty_missions | 9 | ❌ Not Implemented | 0% - Schema only |
| | user_mission_progress | 6 | ❌ Not Implemented | 0% - Schema only |
| | loyalty_points_history | 6 | ✅ Active | Full |
| | badges | 5 | ❌ Not Implemented | 0% - Schema only |
| | user_badges | 3 | ❌ Not Implemented | 0% - Schema only |
| **Locations** | stores | 11 | ✅ Active | Full |
| **Notifications** | notifications | 6 | ✅ Active | Full (auto-created for orders) |
| **Payment** | saved_payment_methods | 6 | ⚠️ Minimal Use | Schema exists, minimal implementation |
| **Admin** | admin_users | 10 | ✅ Active | Full |
| | admin_activity_log | 8 | ✅ Active | Full (audit trail) |

---

## Icon/Image Fields Usage

### Icon Fields
| Table | Field | Type | Current Usage |
|-------|-------|------|----------------|
| categories | `icon_url` | VARCHAR(500) | Defined but not actively used |
| categories | `icon` | VARCHAR(10) | ✅ Used (emoji: ☕ 🥤 🍰 🧋) |
| badges | `icon_url` | VARCHAR(500) | Not implemented |

### Image Fields
| Table | Field | Type | Current Usage |
|-------|-------|------|----------------|
| products | `image_url` | VARCHAR(500) | ✅ Used throughout |
| users | `avatar_url` | VARCHAR(500) | ✅ Used (profile) |
| reviews | `images` | JSON | Schema exists, minimal use |

---

## Key Bugs Found

### Bug #1: Wrong Column Name in cart.py
**File**: `/home/user/Coffee-shop/models/cart.py` Line 58
```sql
-- WRONG ❌
SELECT c.*, p.name as product_name, p.image as product_image

-- CORRECT ✅
SELECT c.*, p.name as product_name, p.image_url as product_image
```
**Impact**: Cart queries will fail at runtime
**Related to branch**: `claude/refactor-db-schema-images-01KcVVKCBuX6H6XuRk8M8fhA`

---

## Unused Features (Database Tables with No Implementation)

### 1. OTP Codes
- **Tables**: otp_codes
- **Issue**: Only prints OTP to console - no email/SMS integration
- **Location**: auth_controller.py line 125: `print(f"OTP Code: {otp_code} for {identifier}")`
- **Status**: Placeholder implementation

### 2. Badges/Achievement System
- **Tables**: badges, user_badges
- **Issue**: NO models, controllers, or views
- **Status**: 0% implemented

### 3. Loyalty Missions/Challenges
- **Tables**: loyalty_missions, user_mission_progress
- **Issue**: NO models, controllers, or views
- **Status**: 0% implemented

### 4. Order Status History
- **Tables**: order_status_history
- **Issue**: Table exists but not actively updated/tracked
- **Status**: Minimal use

---

## Duplicate Tables

### Favorites System
- **Table 1**: `user_favorites` (in schema.sql) - ❌ Deprecated/Unused
- **Table 2**: `favorites` (in schema_updates.sql) - ✅ Currently Used
- **Status**: Remove user_favorites table and consolidate

---

## Implementation Summary

### Fully Implemented (13 features)
1. User authentication & profiles
2. Product catalog & categories
3. Shopping cart
4. Orders & order management
5. Order status tracking
6. Reviews & ratings
7. Toppings/add-ons
8. Vouchers & discounts
9. Loyalty points (earning/redeeming)
10. Store/branch management
11. Notifications (auto-created)
12. Favorites/wishlist
13. Admin functionality (dashboard, management, audit log)

### Partially Implemented (3 features)
1. User preferences (stored but not used for auto-fill)
2. Email/phone verification (flags exist, no service)
3. Payment methods (schema exists, no gateway)

### Not Implemented (4 features)
1. OTP verification (placeholder only)
2. Badges system (schema only)
3. Loyalty missions (schema only)
4. Order status history tracking (schema only)

---

## Database Statistics

- **Total Tables**: 29
- **Total Columns**: ~200+
- **Foreign Keys**: 35+
- **Indexes**: 20+
- **JSON Columns**: 7
- **ENUM Types**: 25+

---

## Files Referenced

- Main schema: `/database/schema.sql`
- Admin schema: `/database/admin_schema.sql`
- Schema updates: `/database/schema_updates.sql`
- Database manager: `/utils/database.py`
- Models directory: `/models/`
- Controllers directory: `/controllers/`

