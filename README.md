# MatrixDNA – Forecasting Assignment

End-to-end pipeline: data cleaning → EDA → forecasting (SARIMAX + regression + FS).

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Put your three CSVs into data/raw/
python -m matrixdna.cli clean --config configs/base.yaml
python -m matrixdna.cli eda   --config configs/base.yaml
python -m matrixdna.cli model --config configs/base.yaml
```

**Outputs**
- Cleaned Excel → `data/processed/matrixdna_cleaned_data.xlsx`
- EDA figures/stats → `reports/eda/`
- Forecast plots/tables → `reports/forecast/`

**Code**
- `src/matrixdna/preprocessing.py`: reads raw CSVs, cleans, writes processed Excel.
- `src/matrixdna/eda.py`: figures & statistical checks for drivers.
- `src/matrixdna/forecasting.py`: baselines (SARIMAX, regression), FS (Ada/ET), what-if.
- `src/matrixdna/cli.py`: tiny CLI to run each step.
