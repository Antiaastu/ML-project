# Starts the Streamlit app in Docker, then open http://localhost:8501
# Usage (PowerShell, from this folder):  .\run_app.ps1
docker run --rm -p 127.0.0.1:8501:8501 -v "${PWD}:/work" ml-classification `
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false
