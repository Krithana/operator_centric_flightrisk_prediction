
# Operator-Centric Flight Risk Prediction

This repository contains code, data, and models for operator-centric aviation safety risk prediction, including explainable AI analysis and reproducible workflows.

## Project Structure

```
.
├── notebooks/                # Jupyter notebooks for data analysis, modeling, and explainability
├── src/                      # Core Python modules for data processing, modeling, and evaluation
├── scripts/                  # Utility scripts for ensemble modeling
├── streamlit/                # Streamlit app utilities and components
├── data/
│   ├── raw/                  # Raw data files (e.g., airline-safety.csv, AviationData.csv)
│   └── processed/            # Processed datasets and results (feature sets, splits, summaries)
├── models/                   # Selected trained model files and their metadata
├── config.py                 # Project configuration
├── requirement.txt           # Python dependencies
├── setup_project.py          # Project setup script
├── Deployed_streamlit_app.pdf# Streamlit app deployment guide
```

## Key Files and Folders

- **notebooks/**: Step-by-step workflows for data cleaning, feature engineering, modeling, and explainability.
- **src/**: Modular code for data loading, preprocessing, feature engineering, model training, evaluation, and more.
- **scripts/**: Scripts for ensemble boosting and quick ensemble runs.
- **streamlit/**: Streamlit app utilities and components.
- **data/raw/**: Original data sources (e.g., `airline-safety.csv`, `AviationData.csv`).
- **data/processed/**: Cleaned and processed datasets, feature importances, splits, and result summaries.
- **models/**: Selected trained models and their metadata.
- **config.py**: Central configuration for paths and settings.
- **requirement.txt**: List of required Python packages.
- **setup_project.py**: Script to set up the project environment.
- **Deployed_streamlit_app.pdf**: Documentation for deploying the Streamlit app.

## How to Reproduce

1. **Install dependencies**
	```bash
	pip install -r requirement.txt
	```

2. **Run notebooks**
	- See the `notebooks/` folder for end-to-end workflows.

3. **Use scripts and modules**
	- Import from `src/` or run scripts in `scripts/` as needed.

4. **Streamlit app**
	- See `streamlit/` for app components. Main app.py is not included.

## Data

- **Raw data**: `data/raw/airline-safety.csv`, `data/raw/AviationData.csv`
- **Processed data**: See `data/processed/` for all key CSVs used in modeling and analysis.

## Models

- **Trained models**: `models/optimized_ensemble_model.pkl`, `models/xgboost_clean_no_leakage.pkl`, etc.
- **Metadata**: Model performance and optimization details in `models/improved_model_metadata.pkl`, `models/optimized_ensemble_metadata.pkl`.

## Results

- **Feature importances**: `data/processed/shap_feature_importance.csv`, `data/processed/comprehensive_feature_importance.csv`
- **Performance summaries**: `data/processed/performance_summary.csv`, `data/processed/validation_summary.csv`
- **Other results**: See `data/processed/` for additional analysis outputs.

### Key Findings
1. Operator safety priors are the strongest predictor
2. Forecast weather indices significantly impact risk
3. Route and airport risk profiles contribute meaningfully
4. Residual 3.17% gap to 80% accuracy reflects inherent booking-stage uncertainty

### Visualizations
All figures are saved in `results/figures/`:
- Confusion matrix
- ROC curve
- Precision-recall curve
- Feature importance
- Temporal analysis
- Business impact analysis

## Notes

- Large files are included only if referenced in main code/notebooks.
- For files too large for GitHub, see instructions in the relevant notebook/script for regeneration.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the repository**:
	```bash
	cd d:\Projects\set_conference\aerospace\2_depsek-flightproject
	```

2. **Create a virtual environment** (recommended):
	```bash
	python -m venv venv
	.\venv\Scripts\activate  # On Windows
	```

3. **Install dependencies**:
	```bash
	pip install -r requirements.txt
	```

## Usage

### Quick Start

Run the complete experiment pipeline:
```bash
python run_experiment.py
```

### Advanced Options

```bash
# Run with specific configuration
python run_experiment.py --mode full --optimize

# Run fast validation only
python run_experiment.py --mode fast
# Generate visualizations only
python run_experiment.py --mode visualize
```

### Using Jupyter Notebooks

Launch Jupyter Lab to explore the analysis notebooks:
```bash
jupyter lab
```

Then open notebooks in order:
1. `01_data_exploration.ipynb` - Exploratory data analysis
2. `02_feature_engineering.ipynb` - Feature engineering
3. `03_model_experiments.ipynb` - Model training experiments
4. `04_model_optimization_production.ipynb` - Final model optimization
5. `05_comprehensive_validation_analysis.ipynb` - Validation and results

## Streamlit App Deployment

To deploy the Streamlit app for operator-centric flight risk prediction:

1. **Install Streamlit (if not already installed):**
	```bash
	pip install streamlit
	```

2. **Navigate to the streamlit directory:**
	```bash
	cd streamlit
	```

3. **Run the Streamlit app:**
	```bash
	streamlit run app.py
	```
	*(If `app.py` is not present, use the appropriate main app file or see `app_backup.py` and components.)*

4. **Access the app:**
	- Open the local URL provided in the terminal (usually http://localhost:8501) in your web browser.

5. **Deployment (optional):**
	- For cloud deployment, see `Deployed_streamlit_app.pdf` for detailed instructions on deploying to Streamlit Cloud or other platforms.

**Note:**
  - Ensure all required model and data files are available in the correct paths as referenced in the app code.
  - Update any file paths in the app configuration if your directory structure differs.

## Reproducibility

### Ensuring Reproducibility
1. **Fixed random seeds**: `RANDOM_STATE = 42` in config.py
2. **Leakage controls**: Booking-timestamp validation
3. **Stratified splitting**: Maintains class distribution
4. **Cross-validation**: 10-fold stratified CV
5. **Documented pipeline**: All steps logged

### Running Tests
```bash
pytest tests/
```

## Acknowledgments

- Data sources: Kaggle aviation datasets
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, pandas, numpy




