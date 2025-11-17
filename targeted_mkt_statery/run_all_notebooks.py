"""
Script để chạy tất cả notebooks theo thứ tự
Run this: python run_all_notebooks.py
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
from pathlib import Path

def run_notebook(notebook_path):
    """Execute a notebook and save the output"""
    print(f"\n{'='*80}")
    print(f"Running: {notebook_path.name}")
    print(f"{'='*80}\n")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure the executor
    ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        
        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"\n✓ {notebook_path.name} completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error executing {notebook_path.name}:")
        print(f"  {str(e)}\n")
        return False

def main():
    # Define notebook order
    notebooks_dir = Path(__file__).parent / 'notebooks'
    notebooks = [
        '01_data_loading_and_eda.ipynb',
        '02_data_preprocessing.ipynb',
        '03_model_training.ipynb',
        '04_model_evaluation.ipynb'
    ]
    
    print("="*80)
    print("RUNNING ALL NOTEBOOKS")
    print("="*80)
    print(f"Total notebooks: {len(notebooks)}")
    print(f"Working directory: {notebooks_dir}")
    print("="*80)
    
    results = {}
    
    for notebook_name in notebooks:
        notebook_path = notebooks_dir / notebook_name
        
        if not notebook_path.exists():
            print(f"✗ Notebook not found: {notebook_name}")
            results[notebook_name] = False
            continue
        
        success = run_notebook(notebook_path)
        results[notebook_name] = success
        
        if not success:
            print(f"\nStopping execution due to error in {notebook_name}")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    for notebook_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12s} - {notebook_name}")
    
    print("="*80)
    
    # Return exit code
    if all(results.values()):
        print("\n🎉 All notebooks executed successfully!")
        return 0
    else:
        print("\n⚠️ Some notebooks failed to execute.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
