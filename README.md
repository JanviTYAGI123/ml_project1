# Medical Insurance Cost Prediction – Streamlit App

Public Streamlit app powered by a trained scikit-learn Pipeline saved via joblib.

## Highlights (for interviews)
- End-to-end: data → preprocessing → model → deployment
- Reproducible training via `train.py` saving a Pipeline with encoders
- Streamlit UI with sidebar, sample input, caching, and dark theme config
- Small, production-friendly artifact (`models/model.joblib`, compressed)

## Local run (Windows PowerShell)
```powershell
cd "C:\Users\JANVI TYAGI\Desktop\insuarance"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ensure your model is at models\model.joblib (or run training below)
streamlit run app.py
```

## Repository structure
```
insuarance/
  app.py
  train.py
  models/
    model.joblib  # <--- add your exported model here
  requirements.txt
  README.md
  .gitignore
  .streamlit/
    config.toml
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a new GitHub repository (Public).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click "New app" → pick your repo and branch → set main file to `app.py` → Deploy.
4. In app settings → Access control → set to "Anyone can view".

## Notes
- The app expects a scikit-learn Pipeline at `models/model.joblib`.
- If your preprocessing is not inside the saved Pipeline, add the same steps in `app.py` before `predict` (or use `train.py` and export a Pipeline).
- Keep model files < 100 MB for reliable GitHub and Streamlit deploys.

## Re-train the model
```powershell
python train.py --csv "insurance (1).csv" --out models\model.joblib
```
Outputs validation MAE and R^2, and writes a compressed `models/model.joblib`.

