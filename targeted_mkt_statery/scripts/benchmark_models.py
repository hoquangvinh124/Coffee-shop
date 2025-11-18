"""
Production Benchmarking Script for ML Models

Measures inference speed, memory usage, and loading time for all trained models
to assess production readiness and performance characteristics.
"""

import os
import sys
import time
import psutil
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Try importing TensorFlow for DNN model
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. DNN model benchmarking will be skipped.")


class ModelBenchmark:
    """Benchmark ML models for production readiness."""
    
    def __init__(self, models_dir: str, data_dir: str):
        """
        Initialize benchmarker.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing test data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results = []
        
    def load_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load test data for benchmarking."""
        print("Loading test data...")
        X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        y_test = pd.read_csv(self.data_dir / 'y_test.csv').values.ravel()
        print(f"Loaded {len(X_test)} test samples with {X_test.shape[1]} features")
        return X_test, y_test
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_loading_time(self, model_path: Path, model_type: str) -> Dict:
        """
        Benchmark model loading time.
        
        Args:
            model_path: Path to model file
            model_type: Type of model (for loading method)
            
        Returns:
            Dictionary with loading metrics
        """
        mem_before = self.get_memory_usage()
        
        start_time = time.time()
        
        if model_type == 'keras' and TF_AVAILABLE:
            model = keras.models.load_model(model_path)
        elif model_type == 'sklearn':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)
        
        load_time = time.time() - start_time
        mem_after = self.get_memory_usage()
        
        return {
            'model': model,
            'load_time_sec': load_time,
            'memory_mb': mem_after - mem_before,
            'total_memory_mb': mem_after
        }
    
    def benchmark_inference_speed(
        self,
        model,
        X_test: pd.DataFrame,
        batch_sizes: List[int] = [1, 10, 100, 1000],
        n_iterations: int = 10
    ) -> Dict:
        """
        Benchmark inference speed for different batch sizes.
        
        Args:
            model: Trained model
            X_test: Test data
            batch_sizes: List of batch sizes to test
            n_iterations: Number of iterations per batch size
            
        Returns:
            Dictionary with inference metrics
        """
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue
            
            times = []
            
            for _ in range(n_iterations):
                # Get random batch
                indices = np.random.choice(len(X_test), batch_size, replace=False)
                X_batch = X_test.iloc[indices]
                
                # Time prediction
                start_time = time.time()
                _ = model.predict(X_batch)
                elapsed = time.time() - start_time
                
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[f'batch_{batch_size}'] = {
                'avg_time_sec': avg_time,
                'std_time_sec': std_time,
                'predictions_per_sec': batch_size / avg_time,
                'time_per_prediction_ms': (avg_time / batch_size) * 1000
            }
        
        return results
    
    def benchmark_model(
        self,
        model_name: str,
        model_path: Path,
        model_type: str,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict:
        """
        Complete benchmark for a single model.
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            model_type: Type of model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with all benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        # 1. Loading time
        print("1. Testing model loading...")
        load_results = self.benchmark_loading_time(model_path, model_type)
        model = load_results['model']
        
        print(f"   Load time: {load_results['load_time_sec']:.4f} sec")
        print(f"   Memory usage: {load_results['memory_mb']:.2f} MB")
        
        # 2. Inference speed
        print("2. Testing inference speed...")
        batch_sizes = [1, 10, 100, 1000, 5000]
        inference_results = self.benchmark_inference_speed(
            model, X_test, batch_sizes, n_iterations=10
        )
        
        for batch_size, metrics in inference_results.items():
            print(f"   {batch_size}: {metrics['predictions_per_sec']:.2f} pred/sec "
                  f"({metrics['time_per_prediction_ms']:.3f} ms/pred)")
        
        # 3. Full dataset prediction
        print("3. Testing full dataset prediction...")
        mem_before = self.get_memory_usage()
        start_time = time.time()
        y_pred = model.predict(X_test)
        full_pred_time = time.time() - start_time
        mem_after = self.get_memory_usage()
        
        print(f"   Full prediction time: {full_pred_time:.4f} sec")
        print(f"   Total samples: {len(X_test)}")
        print(f"   Throughput: {len(X_test) / full_pred_time:.2f} pred/sec")
        print(f"   Memory delta: {mem_after - mem_before:.2f} MB")
        
        # 4. Accuracy check
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1:.4f}")
        
        # Compile results
        result = {
            'model_name': model_name,
            'model_type': model_type,
            'load_time_sec': load_results['load_time_sec'],
            'model_memory_mb': load_results['memory_mb'],
            'total_memory_mb': load_results['total_memory_mb'],
            'full_dataset_time_sec': full_pred_time,
            'full_dataset_throughput': len(X_test) / full_pred_time,
            'prediction_memory_mb': mem_after - mem_before,
            'accuracy': accuracy,
            'f1_weighted': f1,
            **{f'{k}_pred_per_sec': v['predictions_per_sec'] 
               for k, v in inference_results.items()},
            **{f'{k}_ms_per_pred': v['time_per_prediction_ms'] 
               for k, v in inference_results.items()}
        }
        
        return result
    
    def benchmark_all_models(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Benchmark all available models."""
        
        # Define models to benchmark
        models_config = [
            ('XGBoost Standard', 'xgboost_model.pkl', 'joblib'),
            ('XGBoost Resampled', 'xgboost_resampled_model.pkl', 'joblib'),
            ('Random Forest', 'random_forest_model.pkl', 'joblib'),
        ]
        
        # Add DNN if TensorFlow is available
        if TF_AVAILABLE:
            models_config.append(
                ('DNN Entity Embedding', 'dnn_entity_embedding.h5', 'keras')
            )
        
        for model_name, model_file, model_type in models_config:
            model_path = self.models_dir / model_file
            
            if not model_path.exists():
                print(f"\nSkipping {model_name}: File not found at {model_path}")
                continue
            
            try:
                result = self.benchmark_model(
                    model_name, model_path, model_type, X_test, y_test
                )
                self.results.append(result)
            except Exception as e:
                print(f"\nError benchmarking {model_name}: {e}")
                continue
    
    def save_results(self, output_dir: Path):
        """Save benchmark results to CSV and generate report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = output_dir / 'production_benchmarks.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved detailed results to {csv_path}")
        
        # Generate summary report
        self.generate_summary_report(df, output_dir)
    
    def generate_summary_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate human-readable summary report."""
        report_lines = [
            "=" * 80,
            "PRODUCTION READINESS BENCHMARK REPORT",
            "=" * 80,
            "",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Tested: {len(df)}",
            "",
            "=" * 80,
            "1. MODEL LOADING PERFORMANCE",
            "=" * 80,
            ""
        ]
        
        # Loading performance
        loading_summary = df[['model_name', 'load_time_sec', 'model_memory_mb']].copy()
        loading_summary = loading_summary.sort_values('load_time_sec')
        
        for _, row in loading_summary.iterrows():
            report_lines.append(
                f"{row['model_name']:30s}: "
                f"Load={row['load_time_sec']:.4f}s, "
                f"Memory={row['model_memory_mb']:.2f}MB"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "2. INFERENCE SPEED (Batch Size = 1)",
            "=" * 80,
            ""
        ])
        
        # Single prediction speed
        speed_summary = df[['model_name', 'batch_1_ms_per_pred', 'batch_1_pred_per_sec']].copy()
        speed_summary = speed_summary.sort_values('batch_1_ms_per_pred')
        
        for _, row in speed_summary.iterrows():
            report_lines.append(
                f"{row['model_name']:30s}: "
                f"{row['batch_1_ms_per_pred']:.3f}ms/pred "
                f"({row['batch_1_pred_per_sec']:.2f} pred/sec)"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "3. THROUGHPUT (Full Dataset)",
            "=" * 80,
            ""
        ])
        
        # Full dataset throughput
        throughput_summary = df[['model_name', 'full_dataset_throughput', 'full_dataset_time_sec']].copy()
        throughput_summary = throughput_summary.sort_values('full_dataset_throughput', ascending=False)
        
        for _, row in throughput_summary.iterrows():
            report_lines.append(
                f"{row['model_name']:30s}: "
                f"{row['full_dataset_throughput']:.2f} pred/sec "
                f"(Total time: {row['full_dataset_time_sec']:.4f}s)"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "4. ACCURACY vs SPEED TRADE-OFF",
            "=" * 80,
            ""
        ])
        
        # Accuracy vs Speed
        tradeoff = df[['model_name', 'f1_weighted', 'batch_1_pred_per_sec']].copy()
        tradeoff['score'] = tradeoff['f1_weighted'] * tradeoff['batch_1_pred_per_sec']
        tradeoff = tradeoff.sort_values('score', ascending=False)
        
        for _, row in tradeoff.iterrows():
            report_lines.append(
                f"{row['model_name']:30s}: "
                f"F1={row['f1_weighted']:.4f}, "
                f"Speed={row['batch_1_pred_per_sec']:.2f} pred/sec, "
                f"Score={row['score']:.2f}"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "5. PRODUCTION RECOMMENDATIONS",
            "=" * 80,
            ""
        ])
        
        # Recommendations
        fastest = speed_summary.iloc[0]
        most_accurate = df.loc[df['f1_weighted'].idxmax()]
        best_balanced = tradeoff.iloc[0]
        
        report_lines.extend([
            f"Fastest Model: {fastest['model_name']}",
            f"  → Best for: High-volume, latency-critical applications",
            f"  → Speed: {fastest['batch_1_ms_per_pred']:.3f} ms/prediction",
            "",
            f"Most Accurate Model: {most_accurate['model_name']}",
            f"  → Best for: Accuracy-critical decisions",
            f"  → F1-Score: {most_accurate['f1_weighted']:.4f}",
            "",
            f"Best Balanced Model: {best_balanced['model_name']}",
            f"  → Best for: Production deployment (speed + accuracy)",
            f"  → F1: {best_balanced['f1_weighted']:.4f}, Speed: {best_balanced['batch_1_pred_per_sec']:.2f} pred/sec",
            "",
            "Performance Tiers:",
        ])
        
        # Performance tiers
        for _, row in df.iterrows():
            if row['batch_1_pred_per_sec'] > 1000:
                tier = "EXCELLENT"
            elif row['batch_1_pred_per_sec'] > 500:
                tier = "GOOD"
            elif row['batch_1_pred_per_sec'] > 100:
                tier = "ACCEPTABLE"
            else:
                tier = "NEEDS OPTIMIZATION"
            
            report_lines.append(f"  {row['model_name']:30s}: {tier}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        # Save report
        report_path = output_dir / 'production_benchmarks_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Saved summary report to {report_path}")
        
        # Print to console
        print("\n" + '\n'.join(report_lines))


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    data_dir = project_root / 'data' / 'processed'
    output_dir = project_root / 'results' / 'metrics'
    
    print("=" * 80)
    print("PRODUCTION MODEL BENCHMARKING")
    print("=" * 80)
    print(f"\nProject Root: {project_root}")
    print(f"Models Directory: {models_dir}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Initialize benchmarker
    benchmarker = ModelBenchmark(models_dir, data_dir)
    
    # Load test data
    X_test, y_test = benchmarker.load_test_data()
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("STARTING BENCHMARK TESTS")
    print("=" * 80)
    
    benchmarker.benchmark_all_models(X_test, y_test)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    benchmarker.save_results(output_dir)
    
    print("\n✓ Benchmarking complete!")


if __name__ == '__main__':
    main()
