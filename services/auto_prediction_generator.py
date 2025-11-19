"""
Auto Prediction Generator
Automatically runs Prophet models and imports predictions to database
when no data is found
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import DatabaseManager
from revenue_forecasting.predictor import RevenuePredictor

logger = logging.getLogger(__name__)


class AutoPredictionGenerator:
    """
    Automatically generate predictions using Prophet models
    and import to database when needed
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.predictor = RevenuePredictor()
        self.base_dir = Path(__file__).parent.parent / 'revenue_forecasting'
        self.models_dir = self.base_dir / 'ml-models'
        self.results_dir = self.base_dir / 'results'

    def check_data_exists(self) -> Dict[str, int]:
        """Check if predictions exist in database"""
        try:
            overall = self.db.fetch_one("SELECT COUNT(*) as count FROM overall_predictions")
            stores = self.db.fetch_one("SELECT COUNT(*) as count FROM store_metadata")
            store_pred = self.db.fetch_one("SELECT COUNT(*) as count FROM store_predictions")

            return {
                'overall_predictions': overall['count'] if overall else 0,
                'store_metadata': stores['count'] if stores else 0,
                'store_predictions': store_pred['count'] if store_pred else 0
            }
        except Exception as e:
            logger.error(f"Error checking data: {str(e)}")
            return {
                'overall_predictions': 0,
                'store_metadata': 0,
                'store_predictions': 0
            }

    def has_sufficient_data(self) -> bool:
        """Check if database has sufficient prediction data"""
        data = self.check_data_exists()
        return (
            data['overall_predictions'] > 0 and
            data['store_metadata'] > 0
        )

    def generate_overall_predictions(self, days_future: int = 365) -> pd.DataFrame:
        """
        Generate overall system predictions using Prophet model

        Args:
            days_future: Number of days to predict into future

        Returns:
            DataFrame with predictions
        """
        logger.info("Generating overall predictions using Prophet model...")

        try:
            # Load overall model
            model = self.predictor.load_overall_model()

            # Create future dataframe
            # Get last date from historical data or use today
            last_date = datetime.now()

            # Create future dates
            future_dates = pd.DataFrame({
                'ds': pd.date_range(
                    start=last_date,
                    periods=days_future,
                    freq='D'
                )
            })

            # Make predictions
            forecast = model.predict(future_dates)

            # Select relevant columns
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()

            # Add weekly and yearly if available
            if 'weekly' in forecast.columns:
                result['weekly'] = forecast['weekly']
            if 'yearly' in forecast.columns:
                result['yearly'] = forecast['yearly']

            logger.info(f"Generated {len(result)} overall predictions")
            return result

        except Exception as e:
            logger.error(f"Error generating overall predictions: {str(e)}")
            raise

    def generate_store_predictions(
        self,
        store_nbr: int,
        days_future: int = 365
    ) -> pd.DataFrame:
        """
        Generate predictions for a specific store

        Args:
            store_nbr: Store number
            days_future: Number of days to predict

        Returns:
            DataFrame with predictions
        """
        try:
            # Load store model
            model = self.predictor.load_store_model(store_nbr)

            # Create future dates
            last_date = datetime.now()
            future_dates = pd.DataFrame({
                'ds': pd.date_range(
                    start=last_date,
                    periods=days_future,
                    freq='D'
                )
            })

            # Make predictions
            forecast = model.predict(future_dates)

            # Select relevant columns
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result['store_nbr'] = store_nbr

            return result

        except Exception as e:
            logger.error(f"Error generating predictions for store {store_nbr}: {str(e)}")
            return pd.DataFrame()

    def import_overall_predictions(self, df: pd.DataFrame) -> int:
        """Import overall predictions to database"""
        try:
            # Clear existing data
            self.db.execute_query("DELETE FROM overall_predictions")

            insert_query = """
                INSERT INTO overall_predictions
                (ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly, is_historical)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            today = datetime.now().date()
            data = []

            for _, row in df.iterrows():
                forecast_date = pd.to_datetime(row['ds']).date()
                is_historical = forecast_date < today

                data.append((
                    forecast_date,
                    float(row['yhat']),
                    float(row['yhat_lower']) if 'yhat_lower' in row else None,
                    float(row['yhat_upper']) if 'yhat_upper' in row else None,
                    float(row['trend']) if 'trend' in row else None,
                    float(row['weekly']) if 'weekly' in row else None,
                    float(row['yearly']) if 'yearly' in row else None,
                    is_historical
                ))

            # Insert in batches
            batch_size = 1000
            total = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self.db.execute_many(insert_query, batch)
                total += len(batch)

            logger.info(f"Imported {total} overall predictions")
            return total

        except Exception as e:
            logger.error(f"Error importing overall predictions: {str(e)}")
            return 0

    def import_store_metadata(self) -> int:
        """Import store metadata to database"""
        try:
            # Get all stores from predictor
            stores = self.predictor.get_all_stores()

            if not stores:
                logger.warning("No store metadata found")
                return 0

            # Clear existing data
            self.db.execute_query("DELETE FROM store_predictions")
            self.db.execute_query("DELETE FROM store_metadata")

            insert_query = """
                INSERT INTO store_metadata
                (store_nbr, city, state, type, cluster, total_revenue, avg_daily_sales, std_sales, total_transactions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            data = []
            for store in stores:
                data.append((
                    store['store_nbr'],
                    store.get('city', 'Unknown'),
                    store.get('state', 'Unknown'),
                    store.get('type', 'A'),
                    store.get('cluster', 0),
                    float(store.get('total_revenue', 0)),
                    float(store.get('avg_daily_sales', 0)),
                    float(store.get('std_sales', 0)),
                    int(store.get('total_transactions', 0))
                ))

            self.db.execute_many(insert_query, data)
            logger.info(f"Imported {len(data)} stores metadata")
            return len(data)

        except Exception as e:
            logger.error(f"Error importing store metadata: {str(e)}")
            return 0

    def import_store_predictions(self, df: pd.DataFrame) -> int:
        """Import store predictions to database"""
        try:
            if df.empty:
                return 0

            insert_query = """
                INSERT INTO store_predictions
                (store_nbr, ds, yhat, yhat_lower, yhat_upper, is_historical)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            today = datetime.now().date()
            data = []

            for _, row in df.iterrows():
                forecast_date = pd.to_datetime(row['ds']).date()
                is_historical = forecast_date < today

                data.append((
                    int(row['store_nbr']),
                    forecast_date,
                    float(row['yhat']),
                    float(row['yhat_lower']) if 'yhat_lower' in row else None,
                    float(row['yhat_upper']) if 'yhat_upper' in row else None,
                    is_historical
                ))

            # Insert in batches
            batch_size = 1000
            total = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self.db.execute_many(insert_query, batch)
                total += len(batch)

            return total

        except Exception as e:
            logger.error(f"Error importing store predictions: {str(e)}")
            return 0

    def auto_generate_and_import(
        self,
        days_future: int = 365,
        max_stores: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Automatically generate predictions and import to database

        Args:
            days_future: Number of days to predict
            max_stores: Maximum number of stores to process (None = all)

        Returns:
            Dictionary with import statistics
        """
        print("=" * 70)
        print("AUTO PREDICTION GENERATOR")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        stats = {
            'success': False,
            'overall_predictions': 0,
            'store_metadata': 0,
            'store_predictions': 0,
            'stores_processed': 0,
            'errors': []
        }

        try:
            # Step 1: Generate overall predictions
            print("📊 Step 1: Generating overall predictions...")
            overall_df = self.generate_overall_predictions(days_future)
            overall_count = self.import_overall_predictions(overall_df)
            stats['overall_predictions'] = overall_count
            print(f"   ✓ Generated and imported {overall_count} overall predictions\n")

            # Step 2: Import store metadata
            print("🏪 Step 2: Importing store metadata...")
            stores_count = self.import_store_metadata()
            stats['store_metadata'] = stores_count
            print(f"   ✓ Imported {stores_count} stores\n")

            # Step 3: Generate store-specific predictions
            print("🔮 Step 3: Generating store-specific predictions...")

            available_stores = self.predictor.available_stores
            if max_stores:
                available_stores = available_stores[:max_stores]

            print(f"   Processing {len(available_stores)} stores...")

            all_store_predictions = []
            errors = []

            for i, store_nbr in enumerate(available_stores, 1):
                try:
                    print(f"   [{i}/{len(available_stores)}] Store {store_nbr}...", end='\r')

                    store_df = self.generate_store_predictions(store_nbr, days_future)

                    if not store_df.empty:
                        all_store_predictions.append(store_df)
                        stats['stores_processed'] += 1
                    else:
                        errors.append(f"Store {store_nbr}: No predictions generated")

                except Exception as e:
                    error_msg = f"Store {store_nbr}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            # Combine all store predictions
            if all_store_predictions:
                combined_df = pd.concat(all_store_predictions, ignore_index=True)
                store_pred_count = self.import_store_predictions(combined_df)
                stats['store_predictions'] = store_pred_count
                print(f"\n   ✓ Imported {store_pred_count} store predictions")
            else:
                print(f"\n   ⚠️  No store predictions generated")

            if errors:
                stats['errors'] = errors
                print(f"\n   ⚠️  {len(errors)} stores had errors")

            stats['success'] = True

            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"✓ Overall Predictions:  {stats['overall_predictions']:6d}")
            print(f"✓ Store Metadata:       {stats['store_metadata']:6d}")
            print(f"✓ Store Predictions:    {stats['store_predictions']:6d}")
            print(f"✓ Stores Processed:     {stats['stores_processed']:6d}/{len(available_stores)}")

            if errors:
                print(f"\n⚠️  Errors: {len(errors)}")
                for err in errors[:5]:  # Show first 5 errors
                    print(f"   - {err}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more")

            # Verification
            print("\n" + "=" * 70)
            print("VERIFICATION")
            print("=" * 70)

            # Check future predictions
            future_count = self.db.fetch_one("""
                SELECT COUNT(*) as count
                FROM overall_predictions
                WHERE ds >= CURDATE() AND is_historical = FALSE
            """)

            next_7days = self.db.fetch_one("""
                SELECT
                    COUNT(*) as days,
                    SUM(yhat) as total,
                    AVG(yhat) as avg_daily
                FROM overall_predictions
                WHERE ds >= CURDATE() AND ds <= DATE_ADD(CURDATE(), INTERVAL 7 DAY)
            """)

            print(f"Future Predictions:     {future_count['count'] if future_count else 0}")

            if next_7days and next_7days['days'] > 0:
                print(f"\n💰 Next 7 Days Forecast:")
                print(f"   Total: ${next_7days['total']:,.2f}")
                print(f"   Avg:   ${next_7days['avg_daily']:,.2f}/day")

            print("\n" + "=" * 70)
            print("✅ AUTO GENERATION COMPLETED!")
            print("=" * 70)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            error_msg = f"Critical error: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
            import traceback
            traceback.print_exc()

            print("\n" + "=" * 70)
            print("❌ AUTO GENERATION FAILED")
            print("=" * 70)
            print(f"Error: {error_msg}")

        return stats


def main():
    """Main function"""
    generator = AutoPredictionGenerator()

    # Check if data already exists
    if generator.has_sufficient_data():
        print("⚠️  Database already has prediction data!")
        print("\nExisting data:")
        data = generator.check_data_exists()
        for key, value in data.items():
            print(f"  - {key}: {value}")

        response = input("\nDo you want to regenerate all data? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return

    # Run auto generation
    stats = generator.auto_generate_and_import(
        days_future=365,  # 1 year into future
        max_stores=None   # Process all stores (or set to 10 for testing)
    )

    if stats['success']:
        print("\n🎉 Data is ready for AI Agent!")
        print("\nYou can now run:")
        print("  python test_ai_agent.py")
    else:
        print("\n❌ Some errors occurred. Check logs above.")


if __name__ == "__main__":
    main()
