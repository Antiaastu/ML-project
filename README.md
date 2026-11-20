# ML assignment: classification and regression

| Project | Folder | Streamlit entry point |
|---|---|---|
| 1. Student dropout (classification) | `classification/` | `classification/app.py` |
| 2. Building heating and cooling loads (regression) | `Regression/` | `Regression/app.py` |

Each folder has its own notebooks (`01_` to `03_`), trained models (`models/`), report and presentation (`report/`).

## Run an app locally (Docker)

```powershell
cd classification          # or: cd Regression
docker build -t ml-classification .    # ml-regression for the Regression folder
.\run_app.ps1              # then open http://localhost:8501
```

## Deploy on Streamlit Community Cloud

Create one app per project at https://share.streamlit.io with:

- Repository: this repo, branch `main`
- Main file path: `classification/app.py` or `Regression/app.py`
- Advanced settings, Python version: 3.11

Community Cloud installs `requirements.txt` from the app's folder. It is pinned to the versions the models were trained with.
The notebooks need `requirements-notebooks.txt` and `requirements-extra.txt` instead.
