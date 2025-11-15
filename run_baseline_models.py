"""
Train and Evaluate Baseline Models
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from models.baseline_models import BaselineModels
from models.train_test_split import create_time_series_split


def plot_forecasts(train, test, baseline_models, save_path='results/baseline_forecasts.png'):
    """Plot all baseline forecasts"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: All data with train/test split
    axes[0].plot(train.index, train.values, label='Train', linewidth=2, color='blue', alpha=0.7)
    axes[0].plot(test.index, test.values, label='Test (Actual)', linewidth=2, color='black')

    # Plot selected forecasts
    for model_name in ['naive', 'ma_7', 'seasonal_naive', 'sarima']:
        forecast = baseline_models.get_forecast(model_name)
        if forecast is not None:
            axes[0].plot(forecast.index, forecast.values,
                        label=f'{model_name.upper()}', linewidth=2, alpha=0.7)

    axes[0].axvline(x=test.index[0], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Revenue ($)', fontsize=11)
    axes[0].set_title('Baseline Model Forecasts', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)

    # Plot 2: Test period only (zoomed in)
    axes[1].plot(test.index, test.values, label='Actual', linewidth=3,
                color='black', marker='o', markersize=4)

    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, model_name in enumerate(['naive', 'ma_7', 'seasonal_naive', 'arima', 'sarima']):
        forecast = baseline_models.get_forecast(model_name)
        if forecast is not None:
            axes[1].plot(forecast.index, forecast.values,
                        label=f'{model_name.upper()}',
                        linewidth=2, marker='s', markersize=3,
                        alpha=0.7, color=colors[i % len(colors)])

    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Revenue ($)', fontsize=11)
    axes[1].set_title('Test Period Forecasts (Zoomed)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Forecast plot saved to {save_path}")


def plot_metrics_comparison(results, save_path='results/baseline_metrics_comparison.png'):
    """Plot metrics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAPE
    results_sorted = results.sort_values('MAPE')
    axes[0, 0].barh(results_sorted.index, results_sorted['MAPE'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('MAPE (%)', fontsize=11)
    axes[0, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    axes[0, 0].axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Target < 15%')
    axes[0, 0].legend()

    # RMSE
    results_sorted = results.sort_values('RMSE')
    axes[0, 1].barh(results_sorted.index, results_sorted['RMSE'], color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('RMSE ($)', fontsize=11)
    axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='x')
    axes[0, 1].axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Target < $500')
    axes[0, 1].legend()

    # MAE
    results_sorted = results.sort_values('MAE')
    axes[1, 0].barh(results_sorted.index, results_sorted['MAE'], color='seagreen', alpha=0.7)
    axes[1, 0].set_xlabel('MAE ($)', fontsize=11)
    axes[1, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')

    # R²
    results_sorted = results.sort_values('R2', ascending=True)
    axes[1, 1].barh(results_sorted.index, results_sorted['R2'], color='mediumpurple', alpha=0.7)
    axes[1, 1].set_xlabel('R² Score', fontsize=11)
    axes[1, 1].set_title('R-Squared Score', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    axes[1, 1].axvline(x=0.85, color='red', linestyle='--', alpha=0.5, label='Target > 0.85')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics plot saved to {save_path}")


def main():
    print("="*70)
    print(" BASELINE MODELS TRAINING & EVALUATION")
    print("="*70)

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv('data/processed/daily_revenue.csv',
                     index_col='date', parse_dates=True)
    revenue = df['revenue']
    print(f"✓ Loaded {len(revenue)} samples")

    # Split data (80/10/10)
    print("\n[2/4] Creating train/val/test split...")
    train_df, val_df, test_df = create_time_series_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    train = train_df['revenue']
    val = val_df['revenue']
    test = test_df['revenue']

    # For baseline models, combine train+val for final training
    train_combined = pd.concat([train, val])

    # Train baselines
    print("\n[3/4] Training baseline models...")
    baselines = BaselineModels()
    results = baselines.train_all_baselines(train_combined, test)

    # Save results
    results.to_csv('results/baseline_model_results.csv')
    print(f"\n✓ Results saved to results/baseline_model_results.csv")

    # Plot forecasts
    print("\n[4/4] Creating visualizations...")
    plot_forecasts(train_combined, test, baselines)
    plot_metrics_comparison(results)

    # Save summary
    with open('results/baseline_summary.txt', 'w') as f:
        f.write("BASELINE MODELS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training samples: {len(train_combined)}\n")
        f.write(f"Test samples: {len(test)}\n")
        f.write(f"Date range: {train_combined.index.min()} to {test.index.max()}\n\n")
        f.write("Model Performance:\n")
        f.write("-"*70 + "\n")
        f.write(results.to_string())
        f.write("\n\n")

        best_model = results['MAPE'].idxmin()
        f.write(f"🏆 Best Model: {best_model.upper()}\n")
        f.write(f"   MAPE: {results.loc[best_model, 'MAPE']:.2f}%\n")
        f.write(f"   RMSE: ${results.loc[best_model, 'RMSE']:.2f}\n")
        f.write(f"   MAE: ${results.loc[best_model, 'MAE']:.2f}\n")
        f.write(f"   R²: {results.loc[best_model, 'R2']:.4f}\n")

    print(f"✓ Summary saved to results/baseline_summary.txt")

    print("\n" + "="*70)
    print(" BASELINE MODELS COMPLETE")
    print("="*70)

    return baselines, results


if __name__ == "__main__":
    baselines, results = main()
