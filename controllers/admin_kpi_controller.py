"""
Admin KPI Controller
Handle logistics KPI predictions using pretrained model
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add log_model to path
project_root = Path(__file__).parent.parent
log_model_path = project_root / 'log_model'
sys.path.insert(0, str(log_model_path))

from log_model.src.ml.predict import load_model_artifacts, engineer_features, preprocess_new_data


class AdminKPIController:
    """Controller for KPI prediction operations"""
    
    # Singleton pattern for model loading
    _model = None
    _scaler = None
    _encoders = None
    _model_loaded = False
    
    def __init__(self):
        """Initialize controller"""
        self.model_dir = log_model_path / 'models'
        
    def load_model(self):
        """Load model artifacts (only once)"""
        if not AdminKPIController._model_loaded:
            try:
                AdminKPIController._model, AdminKPIController._scaler, AdminKPIController._encoders = \
                    load_model_artifacts(str(self.model_dir))
                AdminKPIController._model_loaded = True
                return True, "Model loaded successfully"
            except Exception as e:
                return False, f"Error loading model: {str(e)}"
        return True, "Model already loaded"
    
    def validate_single_input(self, data):
        """
        Validate single item input data
        
        Parameters:
        -----------
        data : dict
            Dictionary with item features
            
        Returns:
        --------
        tuple : (is_valid, error_message)
        """
        required_fields = [
            'item_id', 'category', 'stock_level', 'reorder_point',
            'reorder_frequency_days', 'lead_time_days', 'daily_demand',
            'demand_std_dev', 'item_popularity_score', 'zone',
            'picking_time_seconds', 'handling_cost_per_unit', 'unit_price',
            'holding_cost_per_unit_day', 'stockout_count_last_month',
            'order_fulfillment_rate', 'total_orders_last_month', 'turnover_ratio',
            'layout_efficiency_score', 'forecasted_demand_next_7d', 'last_restock_date'
        ]
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate numeric ranges
        try:
            if float(data['stock_level']) < 0:
                return False, "Stock level must be >= 0"
            if float(data['reorder_point']) < 0:
                return False, "Reorder point must be >= 0"
            if float(data['daily_demand']) < 0:
                return False, "Daily demand must be >= 0"
            if not 0 <= float(data['order_fulfillment_rate']) <= 1:
                return False, "Order fulfillment rate must be between 0 and 1"
            if not 0 <= float(data['item_popularity_score']) <= 1:
                return False, "Item popularity score must be between 0 and 1"
            if not 0 <= float(data['layout_efficiency_score']) <= 1:
                return False, "Layout efficiency score must be between 0 and 1"
        except ValueError as e:
            return False, f"Invalid numeric value: {str(e)}"
        
        # Validate category
        valid_categories = ['Electronics', 'Groceries', 'Apparel', 'Automotive', 'Pharma']
        if data['category'] not in valid_categories:
            return False, f"Category must be one of: {', '.join(valid_categories)}"
        
        # Validate zone
        valid_zones = ['A', 'B', 'C', 'D']
        if data['zone'] not in valid_zones:
            return False, f"Zone must be one of: {', '.join(valid_zones)}"
        
        # Validate date format
        try:
            datetime.strptime(data['last_restock_date'], '%Y-%m-%d')
        except ValueError:
            return False, "Last restock date must be in format YYYY-MM-DD"
        
        return True, ""
    
    def predict_single_item(self, item_data):
        """
        Predict KPI for a single item
        
        Parameters:
        -----------
        item_data : dict
            Dictionary with item features
            
        Returns:
        --------
        dict : {'success': bool, 'kpi_score': float, 'interpretation': str, 'error': str}
        """
        # Load model if needed
        success, message = self.load_model()
        if not success:
            return {'success': False, 'error': message}
        
        # Validate input
        is_valid, error_msg = self.validate_single_input(item_data)
        if not is_valid:
            return {'success': False, 'error': error_msg}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([item_data])
            
            # Add storage_location_id (required by model but not used in prediction)
            df['storage_location_id'] = 'L00'
            
            # Engineer features
            df = engineer_features(df)
            
            # Preprocess
            X = preprocess_new_data(df, AdminKPIController._scaler, AdminKPIController._encoders)
            
            # Predict
            kpi_score = AdminKPIController._model.predict(X)[0]
            
            # Interpret result
            interpretation = self.interpret_kpi_score(kpi_score)
            
            return {
                'success': True,
                'kpi_score': float(kpi_score),
                'interpretation': interpretation,
                'error': ''
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Prediction error: {str(e)}"}
    
    def predict_batch(self, csv_file_path):
        """
        Predict KPI for multiple items from CSV
        
        Parameters:
        -----------
        csv_file_path : str
            Path to CSV file with item data
            
        Returns:
        --------
        dict : {'success': bool, 'results': pd.DataFrame, 'stats': dict, 'error': str}
        """
        # Load model if needed
        success, message = self.load_model()
        if not success:
            return {'success': False, 'error': message}
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_cols = [
                'item_id', 'category', 'stock_level', 'reorder_point',
                'reorder_frequency_days', 'lead_time_days', 'daily_demand',
                'demand_std_dev', 'item_popularity_score', 'zone',
                'picking_time_seconds', 'handling_cost_per_unit', 'unit_price',
                'holding_cost_per_unit_day', 'stockout_count_last_month',
                'order_fulfillment_rate', 'total_orders_last_month', 'turnover_ratio',
                'layout_efficiency_score', 'forecasted_demand_next_7d', 'last_restock_date'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {'success': False, 'error': f"Missing columns: {', '.join(missing_cols)}"}
            
            # Add storage_location_id if not present
            if 'storage_location_id' not in df.columns:
                df['storage_location_id'] = 'L00'
            
            # Store item IDs
            item_ids = df['item_id'].values
            
            # Engineer features
            df = engineer_features(df)
            
            # Preprocess
            X = preprocess_new_data(df, AdminKPIController._scaler, AdminKPIController._encoders)
            
            # Predict
            predictions = AdminKPIController._model.predict(X)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'item_id': item_ids,
                'predicted_kpi_score': predictions,
                'interpretation': [self.interpret_kpi_score(score) for score in predictions]
            })
            
            # Calculate statistics
            stats = {
                'total_items': len(predictions),
                'mean_kpi': float(predictions.mean()),
                'min_kpi': float(predictions.min()),
                'max_kpi': float(predictions.max()),
                'std_kpi': float(predictions.std()),
                'excellent_count': int(sum(predictions >= 0.7)),
                'good_count': int(sum((predictions >= 0.5) & (predictions < 0.7))),
                'needs_improvement_count': int(sum(predictions < 0.5))
            }
            
            return {
                'success': True,
                'results': results,
                'stats': stats,
                'error': ''
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Batch prediction error: {str(e)}"}
    
    def interpret_kpi_score(self, score):
        """
        Interpret KPI score
        
        Parameters:
        -----------
        score : float
            KPI score (0-1)
            
        Returns:
        --------
        str : Interpretation text
        """
        if score >= 0.7:
            return "✅ Excellent Performance"
        elif score >= 0.5:
            return "⚠️ Good Performance"
        else:
            return "❌ Needs Improvement"
    
    def get_recommendations(self, kpi_score):
        """
        Get recommendations based on KPI score
        
        Parameters:
        -----------
        kpi_score : float
            KPI score (0-1)
            
        Returns:
        --------
        list : List of recommendation strings
        """
        if kpi_score >= 0.7:
            return [
                "✅ Item is performing well",
                "📈 Maintain current inventory levels",
                "🎯 Continue monitoring demand patterns",
                "💡 Consider as a model for other items"
            ]
        elif kpi_score >= 0.5:
            return [
                "⚠️ Room for improvement",
                "📊 Review demand forecasting accuracy",
                "🔄 Optimize reorder points and frequency",
                "⏱️ Reduce picking time if possible",
                "💰 Analyze cost efficiency"
            ]
        else:
            return [
                "❌ Urgent attention required",
                "🚨 High risk of stockouts or overstocking",
                "📉 Review and adjust inventory parameters",
                "🔍 Investigate root causes (demand variability, lead times)",
                "💸 Check if item is cost-effective",
                "🗂️ Consider repositioning in warehouse for better efficiency"
            ]
    
    def get_feature_importance_info(self):
        """
        Get information about important features
        
        Returns:
        --------
        list : List of tuples (feature_name, importance_score, description)
        """
        return [
            ("order_fulfillment_rate", 0.856, "Percentage of orders fulfilled successfully"),
            ("efficiency_composite", 0.798, "Combined score of layout and fulfillment efficiency"),
            ("fulfillment_quality", 0.845, "Quality metric considering fulfillment rate and stockouts"),
            ("turnover_ratio", 0.742, "How quickly inventory is sold and replaced"),
            ("inventory_health", 0.723, "Overall inventory condition based on turnover and fulfillment"),
            ("item_popularity_score", 0.681, "How popular/in-demand the item is"),
            ("demand_supply_balance", 0.654, "Balance between stock coverage and fulfillment"),
            ("picking_efficiency", 0.612, "Efficiency of picking items from warehouse"),
            ("popularity_turnover", 0.598, "Combined metric of popularity and turnover"),
            ("forecast_accuracy", 0.534, "Accuracy of demand forecasting")
        ]
