FROM apache/airflow:2.8.1

USER airflow

# Copy and install extra ML/pipeline dependencies
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
