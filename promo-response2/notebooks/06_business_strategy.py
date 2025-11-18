"""
Business Strategy & Targeting Recommendations
=============================================
Tạo chiến lược targeting và tính toán ROI từ model predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

print("="*80)
print("BUSINESS STRATEGY & TARGETING RECOMMENDATIONS")
print("="*80)

# Load data and model
print("\n1. LOADING MODEL AND DATA...")

# Try to load the best performing model (Logistic Regression had best ROC-AUC)
try:
    model = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    model_name = "Logistic Regression"
except:
    model = joblib.load(MODELS_DIR / "final_best_model.pkl")
    model_name = "LightGBM"

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

# Load original test data for business context
df_full = pd.read_csv(BASE_DIR / "data" / "data_engineered.csv")

# Create proper mapping for test indices
from sklearn.model_selection import train_test_split
X_full = df_full.drop('conversion', axis=1)
y_full = df_full['conversion']
_, X_test_full, _, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
df_test = df_full.loc[X_test_full.index].copy()

print(f"✓ Model loaded: {model_name}")
print(f"✓ Test samples: {len(X_test):,}")

# Make predictions
print("\n2. GENERATING PREDICTIONS...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Add predictions to dataframe
df_test['predicted_proba'] = y_pred_proba
df_test['predicted_class'] = y_pred
df_test['actual_conversion'] = y_test

# Performance metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✓ Model ROC-AUC: {roc_auc:.4f}")

# Business assumptions
print("\n3. BUSINESS ASSUMPTIONS...")
print("-" * 80)

# Revenue assumptions (điều chỉnh theo thực tế quán cafe)
AVG_ORDER_VALUE = 150_000  # VND (trung bình mỗi đơn)
CAMPAIGN_COST_PER_CUSTOMER = 5_000  # VND (chi phí gửi promo/SMS/email)
DISCOUNT_RATE = 0.15  # 15% discount
PROFIT_MARGIN = 0.40  # 40% lợi nhuận trên revenue

print(f"Average order value: {AVG_ORDER_VALUE:,} VND")
print(f"Campaign cost per customer: {CAMPAIGN_COST_PER_CUSTOMER:,} VND")
print(f"Average discount rate: {DISCOUNT_RATE*100:.0f}%")
print(f"Profit margin: {PROFIT_MARGIN*100:.0f}%")

# Calculate business metrics
def calculate_roi(df, threshold=0.5):
    """
    Calculate ROI for a given prediction threshold
    """
    # Select customers to target
    targeted = df[df['predicted_proba'] >= threshold].copy()
    
    # Number of customers targeted
    n_targeted = len(targeted)
    
    # Actual conversions in targeted group
    n_conversions = targeted['actual_conversion'].sum()
    
    # Conversion rate in targeted group
    conversion_rate = n_conversions / n_targeted if n_targeted > 0 else 0
    
    # Revenue
    gross_revenue = n_conversions * AVG_ORDER_VALUE
    discount_amount = gross_revenue * DISCOUNT_RATE
    net_revenue = gross_revenue - discount_amount
    profit = net_revenue * PROFIT_MARGIN
    
    # Costs
    campaign_cost = n_targeted * CAMPAIGN_COST_PER_CUSTOMER
    
    # ROI
    total_profit = profit - campaign_cost
    roi = (total_profit / campaign_cost) if campaign_cost > 0 else 0
    
    return {
        'threshold': threshold,
        'n_targeted': n_targeted,
        'n_conversions': n_conversions,
        'conversion_rate': conversion_rate,
        'gross_revenue': gross_revenue,
        'net_revenue': net_revenue,
        'profit': profit,
        'campaign_cost': campaign_cost,
        'total_profit': total_profit,
        'roi': roi
    }

# Test different thresholds
print("\n4. ROI ANALYSIS BY THRESHOLD...")
print("="*80)

thresholds = np.arange(0.1, 0.9, 0.05)
roi_results = []

for threshold in thresholds:
    result = calculate_roi(df_test, threshold)
    roi_results.append(result)

roi_df = pd.DataFrame(roi_results)

# Find optimal threshold
optimal_idx = roi_df['roi'].idxmax()
optimal_threshold = roi_df.loc[optimal_idx, 'threshold']
optimal_roi = roi_df.loc[optimal_idx, 'roi']

print(f"\n🎯 OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
print(f"   ROI: {optimal_roi:.2f}x ({optimal_roi*100:.0f}% return)")
print(f"   Customers to target: {roi_df.loc[optimal_idx, 'n_targeted']:,.0f}")
print(f"   Expected conversions: {roi_df.loc[optimal_idx, 'n_conversions']:,.0f}")
print(f"   Conversion rate: {roi_df.loc[optimal_idx, 'conversion_rate']*100:.2f}%")
print(f"   Total profit: {roi_df.loc[optimal_idx, 'total_profit']:,.0f} VND")

# Show top 5 thresholds by ROI
print("\n📊 TOP 5 THRESHOLDS BY ROI:")
print("-" * 80)
top_5 = roi_df.nlargest(5, 'roi')[['threshold', 'n_targeted', 'conversion_rate', 'roi', 'total_profit']]
print(top_5.to_string(index=False))

# Customer segmentation strategy
print("\n5. CUSTOMER SEGMENTATION STRATEGY...")
print("="*80)

# Define segments based on prediction probability
def assign_segment(proba):
    if proba >= 0.7:
        return 'Hot Lead'
    elif proba >= 0.5:
        return 'Warm Lead'
    elif proba >= 0.3:
        return 'Cold Lead'
    else:
        return 'No Target'

df_test['segment'] = df_test['predicted_proba'].apply(assign_segment)

# Segment analysis
segment_analysis = df_test.groupby('segment').agg({
    'predicted_proba': ['count', 'mean'],
    'actual_conversion': ['sum', 'mean']
}).round(4)

segment_analysis.columns = ['Count', 'Avg_Proba', 'Conversions', 'Conversion_Rate']
segment_analysis = segment_analysis.sort_values('Avg_Proba', ascending=False)

print("\nCustomer Segments:")
print("-" * 80)
print(segment_analysis)

# Calculate ROI by segment
print("\n💰 ROI BY SEGMENT:")
print("-" * 80)

for segment in ['Hot Lead', 'Warm Lead', 'Cold Lead']:
    segment_df = df_test[df_test['segment'] == segment]
    if len(segment_df) > 0:
        segment_roi = calculate_roi(segment_df, threshold=0)  # threshold=0 means all in segment
        print(f"\n{segment}:")
        print(f"  Customers: {segment_roi['n_targeted']:,.0f}")
        print(f"  Conversions: {segment_roi['n_conversions']:,.0f}")
        print(f"  Conversion rate: {segment_roi['conversion_rate']*100:.2f}%")
        print(f"  ROI: {segment_roi['roi']:.2f}x")
        print(f"  Profit: {segment_roi['total_profit']:,.0f} VND")

# Feature-based targeting strategies
print("\n6. FEATURE-BASED TARGETING STRATEGIES...")
print("="*80)

# Load feature importance
if hasattr(model, 'feature_importances_'):
    feature_names = X_test.columns
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Predictive Features:")
    print("-" * 80)
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Strategy recommendations based on top features
    print("\n📋 TARGETING STRATEGIES BASED ON TOP FEATURES:")
    print("-" * 80)
    
    top_features = feature_importance_df.head(5)['Feature'].tolist()
    
    strategies = []
    
    # Analyze top feature patterns in high-probability predictions
    high_prob_customers = df_test[df_test['predicted_proba'] >= optimal_threshold]
    
    # Strategy 1: Referral-based
    if 'is_referral' in X_test.columns:
        referral_rate = high_prob_customers['is_referral'].mean() if len(high_prob_customers) > 0 else 0
        strategies.append({
            'name': 'Referral Power Campaign',
            'target': 'Customers from referral program',
            'condition': 'is_referral = 1',
            'expected_lift': f"{referral_rate*100:.1f}% of high-probability customers are referrals",
            'action': 'Double rewards for successful referrals'
        })
    
    # Strategy 2: Recency-based
    if 'recency' in df_test.columns:
        avg_recency = high_prob_customers['recency'].mean() if len(high_prob_customers) > 0 else 0
        strategies.append({
            'name': 'Win-Back Recent Customers',
            'target': f'Customers with recency <= {avg_recency:.0f} days',
            'condition': f'recency <= {avg_recency:.0f}',
            'expected_lift': 'Recent customers show higher conversion',
            'action': 'Time-sensitive offers (24-48h validity)'
        })
    
    # Strategy 3: High-value targeting
    if 'is_high_value' in X_test.columns:
        high_value_rate = high_prob_customers['is_high_value'].mean() if len(high_prob_customers) > 0 else 0
        strategies.append({
            'name': 'VIP Appreciation',
            'target': 'High-value customers',
            'condition': 'is_high_value = 1',
            'expected_lift': f"{high_value_rate*100:.1f}% of high-probability customers are high-value",
            'action': 'Premium offers + loyalty points multiplier'
        })
    
    # Strategy 4: Digital channel preference
    if 'is_digital' in X_test.columns:
        digital_rate = high_prob_customers['is_digital'].mean() if len(high_prob_customers) > 0 else 0
        strategies.append({
            'name': 'Digital-First Campaign',
            'target': 'Web/App users',
            'condition': 'is_digital = 1',
            'expected_lift': f"{digital_rate*100:.1f}% of high-probability customers use digital channels",
            'action': 'App-exclusive flash sales + push notifications'
        })
    
    # Strategy 5: Offer matching
    if 'promo_engagement' in X_test.columns:
        avg_engagement = high_prob_customers['promo_engagement'].mean() if len(high_prob_customers) > 0 else 0
        strategies.append({
            'name': 'Personalized Promo Match',
            'target': 'Customers with high promo engagement',
            'condition': f'promo_engagement > {avg_engagement:.2f}',
            'expected_lift': 'Match offer type to historical preference',
            'action': 'Send discount if used_discount=1, BOGO if used_bogo=1'
        })
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\nStrategy {i}: {strategy['name']}")
        print(f"  Target: {strategy['target']}")
        print(f"  Condition: {strategy['condition']}")
        print(f"  Insight: {strategy['expected_lift']}")
        print(f"  Action: {strategy['action']}")

# Projected impact
print("\n7. PROJECTED BUSINESS IMPACT...")
print("="*80)

# Scale to full customer base (assume test set is representative)
TOTAL_CUSTOMERS = 100_000  # Giả định tổng số khách hàng

scale_factor = TOTAL_CUSTOMERS / len(df_test)

optimal_result = roi_df.loc[optimal_idx]
projected_targeted = int(optimal_result['n_targeted'] * scale_factor)
projected_conversions = int(optimal_result['n_conversions'] * scale_factor)
projected_revenue = optimal_result['net_revenue'] * scale_factor
projected_profit = optimal_result['total_profit'] * scale_factor
projected_campaign_cost = optimal_result['campaign_cost'] * scale_factor

print(f"\nAssuming customer base: {TOTAL_CUSTOMERS:,} customers")
print(f"\nPROJECTED MONTHLY IMPACT:")
print(f"  Customers to target: {projected_targeted:,}")
print(f"  Expected conversions: {projected_conversions:,}")
print(f"  Conversion rate: {optimal_result['conversion_rate']*100:.2f}%")
print(f"  Gross revenue: {projected_revenue:,.0f} VND ({projected_revenue/1_000_000:.1f}M)")
print(f"  Campaign cost: {projected_campaign_cost:,.0f} VND ({projected_campaign_cost/1_000_000:.1f}M)")
print(f"  Net profit: {projected_profit:,.0f} VND ({projected_profit/1_000_000:.1f}M)")
print(f"  ROI: {optimal_result['roi']:.2f}x")

# Annual projection
annual_profit = projected_profit * 12
annual_revenue = projected_revenue * 12

print(f"\nPROJECTED ANNUAL IMPACT:")
print(f"  Revenue: {annual_revenue:,.0f} VND ({annual_revenue/1_000_000_000:.1f}B)")
print(f"  Profit: {annual_profit:,.0f} VND ({annual_profit/1_000_000_000:.1f}B)")

# Visualizations
print("\n8. CREATING VISUALIZATIONS...")
print("-" * 80)

# Figure 1: ROI by threshold
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROI curve
axes[0, 0].plot(roi_df['threshold'], roi_df['roi'], linewidth=2, marker='o', color='green')
axes[0, 0].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
axes[0, 0].axhline(y=optimal_roi, color='red', linestyle='--', alpha=0.3)
axes[0, 0].set_xlabel('Prediction Threshold', fontweight='bold')
axes[0, 0].set_ylabel('ROI (x)', fontweight='bold')
axes[0, 0].set_title('ROI by Prediction Threshold', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Conversion rate by threshold
axes[0, 1].plot(roi_df['threshold'], roi_df['conversion_rate']*100, linewidth=2, marker='s', color='blue')
axes[0, 1].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
axes[0, 1].set_xlabel('Prediction Threshold', fontweight='bold')
axes[0, 1].set_ylabel('Conversion Rate (%)', fontweight='bold')
axes[0, 1].set_title('Conversion Rate by Threshold', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Profit by threshold
axes[1, 0].plot(roi_df['threshold'], roi_df['total_profit']/1_000_000, linewidth=2, marker='^', color='purple')
axes[1, 0].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
axes[1, 0].set_xlabel('Prediction Threshold', fontweight='bold')
axes[1, 0].set_ylabel('Total Profit (Million VND)', fontweight='bold')
axes[1, 0].set_title('Profit by Threshold', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Number targeted by threshold
axes[1, 1].plot(roi_df['threshold'], roi_df['n_targeted'], linewidth=2, marker='d', color='orange')
axes[1, 1].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
axes[1, 1].set_xlabel('Prediction Threshold', fontweight='bold')
axes[1, 1].set_ylabel('Customers Targeted', fontweight='bold')
axes[1, 1].set_title('Target Size by Threshold', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "roi_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved: roi_analysis.png")
plt.close()

# Figure 2: Segment performance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Segment distribution
segment_counts = df_test['segment'].value_counts()
colors = {'Hot Lead': '#d62728', 'Warm Lead': '#ff7f0e', 'Cold Lead': '#1f77b4', 'No Target': '#7f7f7f'}
segment_colors = [colors.get(seg, '#7f7f7f') for seg in segment_counts.index]

axes[0].bar(range(len(segment_counts)), segment_counts.values, color=segment_colors)
axes[0].set_xticks(range(len(segment_counts)))
axes[0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
axes[0].set_ylabel('Number of Customers', fontweight='bold')
axes[0].set_title('Customer Distribution by Segment', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(segment_counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Conversion rate by segment
segment_conv = df_test.groupby('segment')['actual_conversion'].mean() * 100
segment_conv = segment_conv.reindex(['Hot Lead', 'Warm Lead', 'Cold Lead', 'No Target'])
segment_colors_ordered = [colors.get(seg, '#7f7f7f') for seg in segment_conv.index]

axes[1].bar(range(len(segment_conv)), segment_conv.values, color=segment_colors_ordered)
axes[1].set_xticks(range(len(segment_conv)))
axes[1].set_xticklabels(segment_conv.index, rotation=45, ha='right')
axes[1].set_ylabel('Conversion Rate (%)', fontweight='bold')
axes[1].set_title('Conversion Rate by Segment', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(segment_conv.values):
    axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "segment_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved: segment_analysis.png")
plt.close()

# Save business strategy document
print("\n9. SAVING BUSINESS STRATEGY DOCUMENT...")
print("-" * 80)

strategy_doc = f"""
BUSINESS STRATEGY & TARGETING RECOMMENDATIONS
==============================================
Generated by ML Model: {type(model).__name__}
Model Performance: ROC-AUC = {roc_auc:.4f}

EXECUTIVE SUMMARY
=================
Using machine learning predictions, we can optimize promotional campaigns
to maximize ROI and increase revenue for the coffee shop.

OPTIMAL TARGETING STRATEGY
===========================
🎯 Recommended Threshold: {optimal_threshold:.2f}
📊 Expected ROI: {optimal_roi:.2f}x ({optimal_roi*100:.0f}% return on investment)
👥 Customers to Target: {roi_df.loc[optimal_idx, 'n_targeted']:,.0f}
✅ Expected Conversions: {roi_df.loc[optimal_idx, 'n_conversions']:,.0f}
📈 Conversion Rate: {roi_df.loc[optimal_idx, 'conversion_rate']*100:.2f}%
💰 Expected Profit: {roi_df.loc[optimal_idx, 'total_profit']:,.0f} VND

CUSTOMER SEGMENTATION
=====================
{segment_analysis.to_string()}

PROJECTED BUSINESS IMPACT (Monthly)
===================================
Assuming {TOTAL_CUSTOMERS:,} total customers:

📞 Customers to Target: {projected_targeted:,}
✅ Expected Conversions: {projected_conversions:,}
💵 Gross Revenue: {projected_revenue:,.0f} VND ({projected_revenue/1_000_000:.1f}M)
💸 Campaign Cost: {projected_campaign_cost:,.0f} VND ({projected_campaign_cost/1_000_000:.1f}M)
💰 Net Profit: {projected_profit:,.0f} VND ({projected_profit/1_000_000:.1f}M)
📊 ROI: {optimal_result['roi']:.2f}x

PROJECTED ANNUAL IMPACT
========================
📅 Annual Revenue: {annual_revenue:,.0f} VND ({annual_revenue/1_000_000_000:.1f} Billion)
💎 Annual Profit: {annual_profit:,.0f} VND ({annual_profit/1_000_000_000:.1f} Billion)

RECOMMENDED TARGETING STRATEGIES
=================================

""" 

# Add strategies if available
if 'strategies' in locals():
    for i, strategy in enumerate(strategies, 1):
        strategy_doc += f"""
Strategy {i}: {strategy['name']}
{'-'*60}
Target Audience: {strategy['target']}
Selection Criteria: {strategy['condition']}
Key Insight: {strategy['expected_lift']}
Recommended Action: {strategy['action']}

"""

strategy_doc += f"""
IMPLEMENTATION ROADMAP
======================

Phase 1: Pilot Test (Week 1-2)
- Test optimal threshold ({optimal_threshold:.2f}) on small subset (10% of target)
- Monitor conversion rates and ROI
- Adjust threshold if needed based on actual results

Phase 2: Segment-Based Campaigns (Week 3-4)
- Launch separate campaigns for Hot/Warm/Cold leads
- Customize messaging and offers per segment
- Track performance by segment

Phase 3: Feature-Based Targeting (Week 5-6)
- Implement top 5 targeting strategies
- Personalize offers based on customer features
- A/B test different offer types

Phase 4: Scale & Optimize (Week 7+)
- Roll out to full customer base
- Continuous monitoring and optimization
- Retrain model quarterly with new data

KEY SUCCESS METRICS
===================
1. Conversion Rate: Target {roi_df.loc[optimal_idx, 'conversion_rate']*100:.2f}%
2. ROI: Target {optimal_roi:.2f}x or higher
3. Customer Lifetime Value: Track increase post-campaign
4. Repeat Purchase Rate: Measure long-term impact
5. Campaign Efficiency: Cost per conversion

RISK MITIGATION
================
1. Over-targeting: Limit campaigns to max 2 per customer/month
2. Discount fatigue: Rotate offer types (BOGO, discount, free item)
3. Channel saturation: Multi-channel approach (SMS, email, app, in-store)
4. Model drift: Monitor performance monthly, retrain quarterly

NEXT ACTIONS
============
☐ Present strategy to management for approval
☐ Set up campaign infrastructure (SMS/email/app notifications)
☐ Prepare creative assets for each customer segment
☐ Configure tracking and analytics dashboard
☐ Launch pilot test with Hot Lead segment
☐ Schedule weekly performance review meetings

TECHNICAL NOTES
===============
- Model: {model_name}
- Test Performance: ROC-AUC {roc_auc:.4f}
- Feature Count: {len(X_test.columns)}
- Prediction Threshold: {optimal_threshold:.2f}
- Model File: logistic_regression.pkl or final_best_model.pkl

CONTACT FOR QUESTIONS
=====================
Data Science Team
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

strategy_path = RESULTS_DIR / "business_strategy.txt"
with open(strategy_path, 'w', encoding='utf-8') as f:
    f.write(strategy_doc)

print(strategy_doc)
print(f"\n✓ Strategy document saved: {strategy_path}")

# Save ROI analysis
roi_df.to_csv(RESULTS_DIR / "roi_analysis.csv", index=False)
print(f"✓ ROI analysis saved: roi_analysis.csv")

# Save segment analysis
segment_analysis.to_csv(RESULTS_DIR / "segment_analysis.csv")
print(f"✓ Segment analysis saved: segment_analysis.csv")

print("\n" + "="*80)
print("✅ BUSINESS STRATEGY COMPLETE!")
print("="*80)
print(f"\n💡 Key Takeaway:")
print(f"   Target {projected_targeted:,} customers at threshold {optimal_threshold:.2f}")
print(f"   Expected profit: {projected_profit/1_000_000:.1f}M VND/month ({annual_profit/1_000_000_000:.1f}B VND/year)")
print(f"   ROI: {optimal_roi:.2f}x")
print(f"\n📁 All results saved to: {RESULTS_DIR}")
