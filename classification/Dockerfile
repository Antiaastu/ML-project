FROM python:3.11-slim

# xgboost needs the OpenMP runtime library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Layer 1: core ML libraries (cached separately so a network drop later
# does not redo this step)
COPY requirements-notebooks.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-notebooks.txt

# Layer 2: app, notebook and slide tools
COPY requirements-extra.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-extra.txt

# Layer 3: pin the app libraries to the exact versions in requirements.txt,
# the same file Streamlit Community Cloud installs, so local and cloud match.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# The project folder is mounted at /work when the container runs,
# so files created inside the container appear on your Windows disk.
EXPOSE 8888 8501
CMD ["bash"]
