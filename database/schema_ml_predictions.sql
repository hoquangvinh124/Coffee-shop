-- =====================================================
-- ML Revenue Forecasting Database Schema
-- Compatible with Prophet model outputs
-- =====================================================

-- Drop existing tables if needed
-- DROP TABLE IF EXISTS prediction_components;
-- DROP TABLE IF EXISTS store_predictions;
-- DROP TABLE IF EXISTS overall_predictions;
-- DROP TABLE IF EXISTS store_metadata;
-- DROP TABLE IF EXISTS model_metrics;

-- =====================================================
-- 1. STORE METADATA
-- =====================================================
CREATE TABLE IF NOT EXISTS store_metadata (
    store_nbr INT PRIMARY KEY,
    city VARCHAR(100),
    state VARCHAR(100),
    type VARCHAR(10),  -- A, B, C, D, E
    cluster INT,
    total_revenue DECIMAL(15, 2),
    avg_daily_sales DECIMAL(12, 2),
    std_sales DECIMAL(12, 2),
    total_transactions BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_type (type),
    INDEX idx_city (city),
    INDEX idx_cluster (cluster)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 2. OVERALL SYSTEM PREDICTIONS (Prophet full forecast)
-- =====================================================
CREATE TABLE IF NOT EXISTS overall_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ds DATE NOT NULL,  -- Date (Prophet standard column name)
    yhat DECIMAL(15, 2) NOT NULL,  -- Predicted value
    yhat_lower DECIMAL(15, 2),  -- Lower confidence bound
    yhat_upper DECIMAL(15, 2),  -- Upper confidence bound
    trend DECIMAL(15, 2),  -- Trend component
    weekly DECIMAL(15, 2),  -- Weekly seasonality
    yearly DECIMAL(15, 2),  -- Yearly seasonality
    is_historical BOOLEAN DEFAULT FALSE,  -- Historical vs future prediction
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_date (ds),
    INDEX idx_date (ds),
    INDEX idx_is_historical (is_historical)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 3. STORE-SPECIFIC PREDICTIONS
-- =====================================================
CREATE TABLE IF NOT EXISTS store_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    store_nbr INT NOT NULL,
    ds DATE NOT NULL,  -- Date
    yhat DECIMAL(15, 2) NOT NULL,  -- Predicted revenue
    yhat_lower DECIMAL(15, 2),  -- Lower confidence bound
    yhat_upper DECIMAL(15, 2),  -- Upper confidence bound
    is_historical BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY unique_store_date (store_nbr, ds),
    FOREIGN KEY (store_nbr) REFERENCES store_metadata(store_nbr) ON DELETE CASCADE,
    INDEX idx_store (store_nbr),
    INDEX idx_date (ds),
    INDEX idx_store_date (store_nbr, ds)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 4. MODEL PERFORMANCE METRICS
-- =====================================================
CREATE TABLE IF NOT EXISTS model_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- 'overall', 'store'
    store_nbr INT NULL,  -- NULL for overall model
    metric_name VARCHAR(50) NOT NULL,  -- 'MAE', 'MAPE', 'RMSE', 'Coverage'
    metric_value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(20),  -- '$', '%', etc.
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_model_type (model_type),
    INDEX idx_store (store_nbr),
    INDEX idx_metric (metric_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 5. YEARLY FORECAST SUMMARY
-- =====================================================
CREATE TABLE IF NOT EXISTS yearly_forecast_summary (
    year INT PRIMARY KEY,
    avg_daily DECIMAL(15, 2),
    total DECIMAL(15, 2),
    std DECIMAL(15, 2),
    total_lower DECIMAL(15, 2),
    total_upper DECIMAL(15, 2),
    total_m DECIMAL(10, 2),  -- Total in millions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_year (year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 6. AI CHAT HISTORY (for AI Agent)
-- =====================================================
CREATE TABLE IF NOT EXISTS ai_chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    user_id INT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    sql_query TEXT,
    query_results JSON,
    execution_time_ms INT,
    is_successful BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session_id (session_id),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_is_successful (is_successful)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 7. AI INSIGHTS & ALERTS
-- =====================================================
CREATE TABLE IF NOT EXISTS ai_insights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    insight_type VARCHAR(50) NOT NULL,  -- 'revenue_alert', 'trend_change', 'recommendation', etc.
    store_nbr INT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',  -- 'info', 'warning', 'critical'
    metadata JSON,  -- Additional data (thresholds, percentages, etc.)
    is_read BOOLEAN DEFAULT FALSE,
    is_dismissed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,

    INDEX idx_store (store_nbr),
    INDEX idx_insight_type (insight_type),
    INDEX idx_severity (severity),
    INDEX idx_created_at (created_at),
    INDEX idx_is_read (is_read),
    INDEX idx_is_dismissed (is_dismissed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: Latest predictions for next 7 days
CREATE OR REPLACE VIEW v_next_7days_forecast AS
SELECT
    ds as forecast_date,
    yhat as predicted_revenue,
    yhat_lower,
    yhat_upper,
    ROUND((yhat_upper - yhat_lower) / yhat * 100, 2) as uncertainty_pct
FROM overall_predictions
WHERE ds >= CURDATE()
  AND ds <= DATE_ADD(CURDATE(), INTERVAL 7 DAY)
  AND is_historical = FALSE
ORDER BY ds;

-- View: Store performance ranking
CREATE OR REPLACE VIEW v_store_performance AS
SELECT
    sm.store_nbr,
    sm.city,
    sm.state,
    sm.type,
    sm.total_revenue,
    sm.avg_daily_sales,
    RANK() OVER (ORDER BY sm.total_revenue DESC) as revenue_rank,
    RANK() OVER (ORDER BY sm.avg_daily_sales DESC) as daily_sales_rank
FROM store_metadata sm
ORDER BY total_revenue DESC;

-- View: Top 10 stores forecast for next 30 days
CREATE OR REPLACE VIEW v_top_stores_forecast AS
SELECT
    sp.store_nbr,
    sm.city,
    sm.type,
    sp.ds as forecast_date,
    sp.yhat as predicted_revenue,
    sm.avg_daily_sales as historical_avg
FROM store_predictions sp
JOIN store_metadata sm ON sp.store_nbr = sm.store_nbr
WHERE sp.ds >= CURDATE()
  AND sp.ds <= DATE_ADD(CURDATE(), INTERVAL 30 DAY)
  AND sm.store_nbr IN (
      SELECT store_nbr
      FROM store_metadata
      ORDER BY total_revenue DESC
      LIMIT 10
  )
ORDER BY sp.store_nbr, sp.ds;

-- View: Monthly revenue summary
CREATE OR REPLACE VIEW v_monthly_forecast AS
SELECT
    YEAR(ds) as year,
    MONTH(ds) as month,
    DATE_FORMAT(ds, '%Y-%m') as year_month,
    COUNT(*) as days,
    SUM(yhat) as total_predicted,
    AVG(yhat) as avg_daily,
    MIN(yhat) as min_daily,
    MAX(yhat) as max_daily,
    STDDEV(yhat) as std_dev
FROM overall_predictions
WHERE is_historical = FALSE
GROUP BY YEAR(ds), MONTH(ds), DATE_FORMAT(ds, '%Y-%m')
ORDER BY year, month;

-- View: Unread insights
CREATE OR REPLACE VIEW v_unread_insights AS
SELECT
    id,
    insight_type,
    store_nbr,
    title,
    description,
    severity,
    created_at
FROM ai_insights
WHERE is_read = FALSE
  AND is_dismissed = FALSE
  AND (expires_at IS NULL OR expires_at > NOW())
ORDER BY
    CASE severity
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'info' THEN 3
        ELSE 4
    END,
    created_at DESC;

-- =====================================================
-- SAMPLE QUERIES FOR AI AGENT
-- =====================================================

/*
-- Total revenue forecast for next week
SELECT SUM(yhat) as total_forecast,
       AVG(yhat) as avg_daily,
       COUNT(*) as days
FROM overall_predictions
WHERE ds >= CURDATE()
  AND ds <= DATE_ADD(CURDATE(), INTERVAL 7 DAY);

-- Top 5 stores by predicted revenue next 30 days
SELECT
    sp.store_nbr,
    sm.city,
    SUM(sp.yhat) as total_predicted,
    AVG(sp.yhat) as avg_daily
FROM store_predictions sp
JOIN store_metadata sm ON sp.store_nbr = sm.store_nbr
WHERE sp.ds >= CURDATE()
  AND sp.ds <= DATE_ADD(CURDATE(), INTERVAL 30 DAY)
GROUP BY sp.store_nbr, sm.city
ORDER BY total_predicted DESC
LIMIT 5;

-- Weekday analysis
SELECT
    DAYNAME(ds) as weekday,
    AVG(yhat) as avg_revenue,
    STDDEV(yhat) as std_dev,
    COUNT(*) as sample_size
FROM overall_predictions
WHERE is_historical = TRUE
GROUP BY DAYNAME(ds), DAYOFWEEK(ds)
ORDER BY DAYOFWEEK(ds);

-- Year over year comparison
SELECT
    yfs.year,
    yfs.total as total_predicted,
    yfs.avg_daily,
    LAG(yfs.total) OVER (ORDER BY yfs.year) as prev_year_total,
    ROUND(((yfs.total - LAG(yfs.total) OVER (ORDER BY yfs.year)) /
           LAG(yfs.total) OVER (ORDER BY yfs.year) * 100), 2) as yoy_growth_pct
FROM yearly_forecast_summary yfs
ORDER BY yfs.year;
*/

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Already created inline above, but here's a summary:
-- - UNIQUE constraints on (store_nbr, ds) combinations
-- - Indexes on frequently queried columns (store_nbr, ds, type, city)
-- - Composite indexes for common query patterns
-- - Foreign keys for referential integrity

-- =====================================================
-- COMPLETED
-- =====================================================
SELECT 'ML Predictions Schema Created Successfully!' as status;
