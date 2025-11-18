"""
Business Evaluator Module for Marketing Campaign ROI Analysis

This module provides tools to evaluate machine learning models from a business perspective,
calculating ROI, simulating marketing campaigns, and comparing different strategies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional


class BusinessEvaluator:
    """
    Evaluates ML models using business metrics like ROI, cost-benefit analysis,
    and campaign simulation.
    """
    
    def __init__(
        self,
        offer_cost: float = 2.0,
        completion_revenue: float = 10.0,
        view_value: float = 0.5,
        transaction_value: float = 5.0
    ):
        """
        Initialize business evaluator with cost/revenue parameters.
        
        Args:
            offer_cost: Cost to send one offer ($)
            completion_revenue: Revenue from completed offer ($)
            view_value: Value of an offer view/engagement ($)
            transaction_value: Average transaction value ($)
        """
        self.offer_cost = offer_cost
        self.completion_revenue = completion_revenue
        self.view_value = view_value
        self.transaction_value = transaction_value
        
        # Class names mapping
        self.class_names = {
            0: 'Offer Received',
            1: 'Offer Viewed',
            2: 'Transaction',
            3: 'Offer Completed'
        }
        
        # Value per class (negative for cost, positive for revenue)
        self.class_values = {
            0: -offer_cost,  # Cost to send offer
            1: view_value,    # Small value for engagement
            2: transaction_value,  # Transaction revenue
            3: completion_revenue  # High value for completion
        }
    
    def calculate_roi(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_details: bool = False
    ) -> Dict:
        """
        Calculate ROI based on predictions vs actual outcomes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            return_details: If True, return detailed breakdown
            
        Returns:
            Dictionary with ROI metrics
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        n_classes = len(self.class_names)
        
        # Initialize cost/revenue tracking
        total_cost = 0
        total_revenue = 0
        class_breakdown = {}
        
        for true_class in range(n_classes):
            for pred_class in range(n_classes):
                count = cm[true_class, pred_class]
                
                if count == 0:
                    continue
                
                # Cost: We send offers based on prediction
                if pred_class in [0, 1, 3]:  # Offer-related predictions
                    total_cost += count * self.offer_cost
                
                # Revenue: We get value based on actual outcome
                actual_value = self.class_values.get(true_class, 0)
                if actual_value > 0:
                    total_revenue += count * actual_value
                
                # Track per-class performance
                key = f"{self.class_names[true_class]} (pred: {self.class_names[pred_class]})"
                class_breakdown[key] = {
                    'count': int(count),
                    'cost': count * (self.offer_cost if pred_class in [0, 1, 3] else 0),
                    'revenue': count * max(actual_value, 0)
                }
        
        # Calculate metrics
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        
        results = {
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'net_profit': net_profit,
            'roi_percentage': roi,
            'n_predictions': len(y_pred),
            'cost_per_prediction': total_cost / len(y_pred) if len(y_pred) > 0 else 0,
            'revenue_per_prediction': total_revenue / len(y_pred) if len(y_pred) > 0 else 0
        }
        
        if return_details:
            results['class_breakdown'] = class_breakdown
        
        return results
    
    def simulate_campaign(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model,
        budget: float,
        strategy: str = 'top_probability',
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Simulate a marketing campaign with budget constraints.
        
        Args:
            X_test: Test features
            y_test: True labels
            model: Trained model with predict_proba method
            budget: Campaign budget ($)
            strategy: 'top_probability', 'threshold', or 'random'
            target_class: Target class for campaign (e.g., 3 for Offer Completed)
            
        Returns:
            Campaign results dictionary
        """
        # Get predictions and probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_proba = None
        
        # Calculate max customers we can target
        max_customers = int(budget / self.offer_cost)
        max_customers = min(max_customers, len(X_test))
        
        # Select customers based on strategy
        if strategy == 'top_probability' and y_proba is not None:
            if target_class is not None:
                # Target specific class
                scores = y_proba[:, target_class]
            else:
                # Target best predicted class per customer
                scores = np.max(y_proba, axis=1)
            
            # Select top N customers
            top_indices = np.argsort(scores)[-max_customers:]
            
        elif strategy == 'threshold' and y_proba is not None:
            # Use probability threshold
            threshold = 0.5
            if target_class is not None:
                mask = y_proba[:, target_class] > threshold
            else:
                mask = np.max(y_proba, axis=1) > threshold
            
            top_indices = np.where(mask)[0][:max_customers]
            
        else:  # random
            top_indices = np.random.choice(len(X_test), max_customers, replace=False)
        
        # Calculate results for selected customers
        selected_true = y_test[top_indices]
        selected_pred = y_pred[top_indices]
        
        # Calculate ROI for campaign
        roi_results = self.calculate_roi(selected_true, selected_pred, return_details=True)
        
        # Add campaign-specific metrics
        roi_results['strategy'] = strategy
        roi_results['target_class'] = target_class
        roi_results['budget'] = budget
        roi_results['customers_targeted'] = len(top_indices)
        roi_results['actual_cost'] = len(top_indices) * self.offer_cost
        roi_results['budget_utilized'] = (len(top_indices) * self.offer_cost / budget * 100)
        
        # Calculate conversion rates
        if target_class is not None:
            conversions = np.sum(selected_true == target_class)
            roi_results['conversion_rate'] = conversions / len(top_indices) * 100
            roi_results['conversions'] = int(conversions)
        
        return roi_results
    
    def compare_strategies(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        models: Dict,
        budget: float = 10000,
        strategies: List[str] = None,
        target_classes: List[int] = None
    ) -> pd.DataFrame:
        """
        Compare different models and strategies.
        
        Args:
            X_test: Test features
            y_test: True labels
            models: Dictionary of {model_name: model}
            budget: Campaign budget
            strategies: List of strategies to test
            target_classes: List of target classes to test
            
        Returns:
            DataFrame with comparison results
        """
        if strategies is None:
            strategies = ['top_probability', 'random']
        
        if target_classes is None:
            target_classes = [None, 3]  # None = best class, 3 = Offer Completed
        
        results = []
        
        for model_name, model in models.items():
            for strategy in strategies:
                for target_class in target_classes:
                    try:
                        campaign_result = self.simulate_campaign(
                            X_test, y_test, model, budget, strategy, target_class
                        )
                        
                        results.append({
                            'model': model_name,
                            'strategy': strategy,
                            'target_class': self.class_names.get(target_class, 'Best') if target_class else 'Best',
                            'roi_%': campaign_result['roi_percentage'],
                            'net_profit': campaign_result['net_profit'],
                            'total_revenue': campaign_result['total_revenue'],
                            'total_cost': campaign_result['total_cost'],
                            'customers_targeted': campaign_result['customers_targeted'],
                            'conversion_rate_%': campaign_result.get('conversion_rate', 0),
                            'budget_utilized_%': campaign_result['budget_utilized']
                        })
                    except Exception as e:
                        print(f"Error with {model_name}, {strategy}, {target_class}: {e}")
                        continue
        
        return pd.DataFrame(results).sort_values('roi_%', ascending=False)
    
    def plot_roi_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot ROI comparison across models and strategies.
        
        Args:
            comparison_df: DataFrame from compare_strategies()
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROI by Model
        ax1 = axes[0, 0]
        model_roi = comparison_df.groupby('model')['roi_%'].mean().sort_values(ascending=False)
        model_roi.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Average ROI by Model', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('ROI (%)')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Net Profit by Model
        ax2 = axes[0, 1]
        model_profit = comparison_df.groupby('model')['net_profit'].mean().sort_values(ascending=False)
        model_profit.plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
        ax2.set_title('Average Net Profit by Model', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Net Profit ($)')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. ROI by Strategy
        ax3 = axes[1, 0]
        strategy_roi = comparison_df.groupby('strategy')['roi_%'].mean().sort_values(ascending=False)
        strategy_roi.plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='black')
        ax3.set_title('Average ROI by Strategy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('ROI (%)')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Conversion Rate vs ROI
        ax4 = axes[1, 1]
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]
            ax4.scatter(
                model_data['conversion_rate_%'],
                model_data['roi_%'],
                label=model,
                s=100,
                alpha=0.6
            )
        ax4.set_title('Conversion Rate vs ROI', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Conversion Rate (%)')
        ax4.set_ylabel('ROI (%)')
        ax4.legend(loc='best')
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROI comparison plot to {save_path}")
        
        plt.show()
    
    def generate_business_report(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive business report.
        
        Args:
            comparison_df: DataFrame from compare_strategies()
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "BUSINESS EVALUATION REPORT - TARGETED MARKETING MODELS",
            "=" * 80,
            "",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Scenarios Tested: {len(comparison_df)}",
            "",
            "=" * 80,
            "1. BEST PERFORMING MODEL",
            "=" * 80,
            ""
        ]
        
        # Best overall model
        best_model = comparison_df.iloc[0]
        report_lines.extend([
            f"Model: {best_model['model']}",
            f"Strategy: {best_model['strategy']}",
            f"Target Class: {best_model['target_class']}",
            f"ROI: {best_model['roi_%']:.2f}%",
            f"Net Profit: ${best_model['net_profit']:,.2f}",
            f"Total Revenue: ${best_model['total_revenue']:,.2f}",
            f"Total Cost: ${best_model['total_cost']:,.2f}",
            f"Customers Targeted: {best_model['customers_targeted']:,}",
            f"Conversion Rate: {best_model['conversion_rate_%']:.2f}%",
            "",
            "=" * 80,
            "2. MODEL RANKINGS BY ROI",
            "=" * 80,
            ""
        ])
        
        # Model rankings
        model_summary = comparison_df.groupby('model').agg({
            'roi_%': 'mean',
            'net_profit': 'mean',
            'conversion_rate_%': 'mean'
        }).sort_values('roi_%', ascending=False)
        
        for idx, (model_name, row) in enumerate(model_summary.iterrows(), 1):
            report_lines.append(
                f"{idx}. {model_name}: "
                f"ROI={row['roi_%']:.2f}%, "
                f"Profit=${row['net_profit']:,.2f}, "
                f"Conv={row['conversion_rate_%']:.2f}%"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "3. STRATEGY COMPARISON",
            "=" * 80,
            ""
        ])
        
        # Strategy comparison
        strategy_summary = comparison_df.groupby('strategy').agg({
            'roi_%': 'mean',
            'net_profit': 'mean'
        }).sort_values('roi_%', ascending=False)
        
        for strategy, row in strategy_summary.iterrows():
            report_lines.append(
                f"{strategy}: ROI={row['roi_%']:.2f}%, Profit=${row['net_profit']:,.2f}"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "4. RECOMMENDATIONS",
            "=" * 80,
            ""
        ])
        
        # Generate recommendations
        best_roi_model = model_summary.index[0]
        best_roi = model_summary.iloc[0]['roi_%']
        
        if best_roi > 100:
            recommendation = "STRONGLY RECOMMENDED for deployment - High positive ROI"
        elif best_roi > 50:
            recommendation = "RECOMMENDED for deployment - Positive ROI"
        elif best_roi > 0:
            recommendation = "CAUTIOUSLY RECOMMENDED - Low but positive ROI"
        else:
            recommendation = "NOT RECOMMENDED - Negative ROI"
        
        report_lines.extend([
            f"Best Model for Deployment: {best_roi_model}",
            f"Expected ROI: {best_roi:.2f}%",
            f"Recommendation: {recommendation}",
            "",
            "Key Insights:",
            f"- Top strategy: {comparison_df.iloc[0]['strategy']}",
            f"- Optimal target: {comparison_df.iloc[0]['target_class']}",
            f"- Average profit per campaign: ${comparison_df['net_profit'].mean():,.2f}",
            "",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Saved business report to {save_path}")
        
        return report_text


# Utility functions
def calculate_class_costs(y_true, y_pred, cost_matrix: Dict[Tuple[int, int], float]) -> float:
    """
    Calculate total cost based on custom cost matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_matrix: Dict mapping (true_class, pred_class) -> cost
        
    Returns:
        Total cost
    """
    cm = confusion_matrix(y_true, y_pred)
    total_cost = 0
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cost = cost_matrix.get((i, j), 0)
            total_cost += cm[i, j] * cost
    
    return total_cost


def optimal_threshold_search(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_class: int,
    evaluator: BusinessEvaluator,
    thresholds: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Find optimal probability threshold for maximum ROI.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_class: Target class to optimize
        evaluator: BusinessEvaluator instance
        thresholds: Array of thresholds to test
        
    Returns:
        Tuple of (best_threshold, best_metrics)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    best_roi = -np.inf
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        # Convert probabilities to predictions using threshold
        y_pred = (y_proba[:, target_class] >= threshold).astype(int) * target_class
        
        # Calculate ROI
        roi_metrics = evaluator.calculate_roi(y_true, y_pred)
        
        if roi_metrics['roi_percentage'] > best_roi:
            best_roi = roi_metrics['roi_percentage']
            best_threshold = threshold
            best_metrics = roi_metrics.copy()
            best_metrics['threshold'] = threshold
    
    return best_threshold, best_metrics
