# Run All Model Testing - Comprehensive Test Suite
# This script executes all model testing components

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   TARGETED MARKETING STRATEGY - COMPREHENSIVE MODEL TESTING" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
$venvPath = "..\.venv\Scripts\Activate.ps1"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Error: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please run setup first or activate your Python environment" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& $venvPath

Write-Host "Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Check if required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Green
$requiredPackages = @(
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "joblib",
    "psutil"
)

$missingPackages = @()
foreach ($package in $requiredPackages) {
    $installed = python -c "import $package" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "Missing packages detected: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "Installing missing packages..." -ForegroundColor Yellow
    pip install $missingPackages
    Write-Host ""
}

Write-Host "All required packages are available" -ForegroundColor Green
Write-Host ""

# Menu for user selection
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SELECT TESTING OPTION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run Production Benchmarking (Fast - 2-5 minutes)" -ForegroundColor White
Write-Host "   - Measures inference speed, memory usage, and loading time" -ForegroundColor Gray
Write-Host "   - Generates production readiness report" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Open Comprehensive Testing Notebook (Interactive)" -ForegroundColor White
Write-Host "   - ROC-AUC curves, Precision-Recall curves" -ForegroundColor Gray
Write-Host "   - Business ROI analysis and campaign simulation" -ForegroundColor Gray
Write-Host "   - Error analysis with SHAP explanations" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Open Temporal Validation Notebook (Interactive)" -ForegroundColor White
Write-Host "   - Time-based validation (train on old, test on new data)" -ForegroundColor Gray
Write-Host "   - Performance degradation analysis" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Run All Tests (Automated - for scripts only)" -ForegroundColor White
Write-Host "   - Runs production benchmarking" -ForegroundColor Gray
Write-Host "   - Opens both notebooks for manual execution" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Generate Testing Report Summary" -ForegroundColor White
Write-Host "   - Aggregates all existing test results" -ForegroundColor Gray
Write-Host ""
Write-Host "0. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (0-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "   RUNNING PRODUCTION BENCHMARKING" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        if (-Not (Test-Path "models\xgboost_model.pkl")) {
            Write-Host "Error: Models not found in models/ directory" -ForegroundColor Red
            Write-Host "Please train models first using notebooks 01-03" -ForegroundColor Yellow
            exit 1
        }
        
        python scripts\benchmark_models.py
        
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Green
        Write-Host "   BENCHMARKING COMPLETE" -ForegroundColor Green
        Write-Host "======================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Results saved to:" -ForegroundColor Green
        Write-Host "  - results/metrics/production_benchmarks.csv" -ForegroundColor White
        Write-Host "  - results/metrics/production_benchmarks_report.txt" -ForegroundColor White
        Write-Host ""
        
        # Ask if user wants to view report
        $viewReport = Read-Host "View report now? (y/n)"
        if ($viewReport -eq "y") {
            Get-Content "results\metrics\production_benchmarks_report.txt"
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "   OPENING COMPREHENSIVE TESTING NOTEBOOK" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        if (-Not (Test-Path "notebooks\05_comprehensive_model_testing.ipynb")) {
            Write-Host "Error: Notebook not found" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Launching Jupyter Notebook..." -ForegroundColor Green
        Write-Host "Please execute all cells in the notebook manually" -ForegroundColor Yellow
        Write-Host ""
        
        jupyter notebook notebooks\05_comprehensive_model_testing.ipynb
    }
    
    "3" {
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "   OPENING TEMPORAL VALIDATION NOTEBOOK" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        if (-Not (Test-Path "notebooks\06_temporal_validation.ipynb")) {
            Write-Host "Error: Notebook not found" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Launching Jupyter Notebook..." -ForegroundColor Green
        Write-Host "Please execute all cells in the notebook manually" -ForegroundColor Yellow
        Write-Host ""
        
        jupyter notebook notebooks\06_temporal_validation.ipynb
    }
    
    "4" {
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "   RUNNING ALL TESTS" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Run benchmarking
        Write-Host "Step 1/3: Running production benchmarking..." -ForegroundColor Green
        python scripts\benchmark_models.py
        Write-Host "Production benchmarking complete!" -ForegroundColor Green
        Write-Host ""
        
        # Open comprehensive testing
        Write-Host "Step 2/3: Opening comprehensive testing notebook..." -ForegroundColor Green
        Write-Host "Please execute all cells and close when done" -ForegroundColor Yellow
        Start-Process "jupyter" -ArgumentList "notebook", "notebooks\05_comprehensive_model_testing.ipynb" -NoNewWindow
        Write-Host ""
        
        # Open temporal validation
        Write-Host "Step 3/3: Opening temporal validation notebook..." -ForegroundColor Green
        Write-Host "Please execute all cells and close when done" -ForegroundColor Yellow
        Start-Process "jupyter" -ArgumentList "notebook", "notebooks\06_temporal_validation.ipynb" -NoNewWindow
        Write-Host ""
        
        Write-Host "All test scripts launched!" -ForegroundColor Green
        Write-Host "Execute notebook cells manually to complete testing" -ForegroundColor Yellow
    }
    
    "5" {
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "   GENERATING TESTING REPORT SUMMARY" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        $reportFiles = @(
            "results\metrics\comprehensive_testing_summary.txt",
            "results\metrics\temporal_validation_summary.txt",
            "results\metrics\production_benchmarks_report.txt"
        )
        
        $summaryPath = "results\metrics\COMPLETE_TESTING_SUMMARY.txt"
        
        "======================================================================" | Out-File $summaryPath
        "   COMPLETE TESTING SUMMARY - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $summaryPath -Append
        "======================================================================" | Out-File $summaryPath -Append
        "" | Out-File $summaryPath -Append
        
        foreach ($file in $reportFiles) {
            if (Test-Path $file) {
                $fileName = Split-Path $file -Leaf
                "----------------------------------------------------------------------" | Out-File $summaryPath -Append
                "   $fileName" | Out-File $summaryPath -Append
                "----------------------------------------------------------------------" | Out-File $summaryPath -Append
                "" | Out-File $summaryPath -Append
                Get-Content $file | Out-File $summaryPath -Append
                "" | Out-File $summaryPath -Append
            } else {
                Write-Host "Warning: $file not found (run tests first)" -ForegroundColor Yellow
            }
        }
        
        "======================================================================" | Out-File $summaryPath -Append
        "   END OF SUMMARY" | Out-File $summaryPath -Append
        "======================================================================" | Out-File $summaryPath -Append
        
        Write-Host "Summary report generated!" -ForegroundColor Green
        Write-Host "Saved to: $summaryPath" -ForegroundColor White
        Write-Host ""
        
        $viewSummary = Read-Host "View summary now? (y/n)"
        if ($viewSummary -eq "y") {
            Get-Content $summaryPath | more
        }
    }
    
    "0" {
        Write-Host ""
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host ""
        Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   TESTING COMPLETE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "For detailed guidance, see TESTING_GUIDE.md" -ForegroundColor Green
Write-Host ""

# Pause before exit
Read-Host "Press Enter to exit"
