FROM apache/airflow:2.9.3-python3.11

USER airflow

# Copy and install extra ML/pipeline dependencies
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
