# Local development image: Airflow 3.4 (Python 3.14) plus an isolated RevIntel environment.
#
# RevIntel's pinned dependencies (requirements/lock.txt) live in their own virtualenv at
# /opt/revintel-venv, so Airflow's constraints and RevIntel's pins never mix. Pipeline
# tasks run there via @task.external_python. RevIntel source code is bind-mounted at
# /opt/revintel/src (see docker-compose.yaml) and put on that venv's path by a .pth file.
FROM apache/airflow:3.4.0-python3.14

USER root
# libgomp1: OpenMP runtime for XGBoost. git: lets the pipeline record the commit it ran.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /opt/revintel/src

COPY lock.txt /tmp/revintel-lock.txt
RUN python -m venv /opt/revintel-venv \
    && /opt/revintel-venv/bin/pip install --no-cache-dir -r /tmp/revintel-lock.txt \
    && site_packages="$(/opt/revintel-venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && printf '/opt/revintel/src\n/opt/revintel/src/backend\n' > "${site_packages}/revintel.pth" \
    && chown -R airflow:0 /opt/revintel-venv \
    && rm /tmp/revintel-lock.txt

USER airflow
# Test runner for the DAG tests, in Airflow's own environment only.
RUN pip install --no-cache-dir pytest==9.1.1
