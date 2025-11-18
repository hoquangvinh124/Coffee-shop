# 📁 Project Structure - Logistics KPI Prediction System

## 🌳 Complete Directory Tree

```
log_model/
│
├── 📂 src/                              # 🎯 Source Code (Modular Architecture)
│   ├── __init__.py                     # Package initializer
│   │
│   ├── 📂 api/                         # REST API Module
│   │   ├── __init__.py
│   │   └── app.py                      # FastAPI server (555 lines)
│   │
│   ├── 📂 dashboard/                   # Dashboard Module
│   │   ├── __init__.py
│   │   └── dashboard.py                # Streamlit dashboard (553 lines)
│   │
│   ├── 📂 ml/                          # Machine Learning Module
│   │   ├── __init__.py
│   │   ├── train_model.py             # Training pipeline (561 lines)
│   │   ├── predict.py                 # Prediction module (316 lines)
│   │   └── hyperparameter_tuning.py   # Optuna-based tuning
│   │
│   └── 📂 utils/                       # Utilities Module
│       ├── __init__.py
│       └── monitoring.py              # Monitoring system (467 lines)
│
├── 📂 config/                          # ⚙️ Configuration
│   ├── __init__.py                    # Config loader
│   └── config.yaml                    # Application settings
│
├── 📂 tests/                           # 🧪 Unit Tests
│   └── test_model.py                  # 13 tests (100% passing)
│
├── 📂 notebooks/                       # 📓 Jupyter Notebooks
│   └── exploratory_data_analysis.ipynb # EDA notebook
│
├── 📂 deployment/                      # 🐳 Deployment Configuration
│   ├── Dockerfile                     # Docker image definition
│   ├── docker-compose.yml             # Multi-container orchestration
│   └── requirements.txt               # Python dependencies
│
├── 📂 scripts/                         # 🔧 Management Scripts
│   ├── startup.bat                    # Start all services (148 lines)
│   ├── shutdown.bat                   # Stop all services (95 lines)
│   ├── status.bat                     # Check system status (102 lines)
│   ├── restart.bat                    # Quick restart (33 lines)
│   └── README.md                      # Scripts documentation
│
├── 📂 data/                            # 📊 Datasets
│   └── logistics_dataset.csv          # Original dataset (3,204 samples)
│
├── 📂 models/                          # 🤖 Trained Models
│   ├── Ridge_Regression_*.pkl         # Best model (R²=99.99%)
│   ├── scaler_*.pkl                   # Feature scaler
│   └── encoders_*.pkl                 # Categorical encoders
│
├── 📂 logs/                            # 📝 Log Files
│   ├── api.log                        # API server logs
│   ├── dashboard.log                  # Dashboard logs
│   └── README.md                      # Logging documentation
│
├── 📂 backups/                         # 💾 Backup Storage
│   └── README.md                      # Backup strategy
│
├── 📂 docs/                            # 📚 Documentation
│   ├── README.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── MONITORING_GUIDE.md
│   ├── QUICK_START.md
│   └── .archive/
│
├── 📄 README.md                        # 🏠 Project README
├── 📄 setup.py                         # 📦 Package Setup
├── 📄 pyproject.toml                   # ⚙️ Project Configuration
├── 📄 Makefile                         # 🛠️ Build Automation
├── 📄 MANIFEST.in                      # 📋 Package Manifest
└── 📄 .gitignore                       # 🚫 Git Ignore Rules
```

---

## 🏗️ Architecture Overview

### Modular Design

- **src/api/** - REST API endpoints (FastAPI)
- **src/dashboard/** - Web UI (Streamlit)
- **src/ml/** - ML pipeline (training, prediction, tuning)
- **src/utils/** - Shared utilities (monitoring, logging)
- **config/** - Centralized configuration (YAML)
- **tests/** - Unit tests (pytest)
- **deployment/** - Docker & requirements
- **scripts/** - Automation scripts

### Clean Separation of Concerns

✅ API logic separated from ML logic  
✅ Dashboard independent from API  
✅ Configuration externalized  
✅ Reusable utility modules  
✅ Comprehensive testing

---

## 📊 Quick Reference

**Start Services**: `scripts\startup.bat`  
**Stop Services**: `scripts\shutdown.bat`  
**Check Status**: `scripts\status.bat`  
**API Docs**: http://localhost:8000/docs  
**Dashboard**: http://localhost:8501

---

<div align="center">

**Clean • Organized • Production-Ready**

</div>
