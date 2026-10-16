CREATE DATABASE revintel_test;
CREATE DATABASE mlflow;
CREATE DATABASE airflow;

\connect revintel
CREATE EXTENSION IF NOT EXISTS vector;

\connect revintel_test
CREATE EXTENSION IF NOT EXISTS vector;