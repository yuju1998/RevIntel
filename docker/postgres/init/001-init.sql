CREATE DATABASE revintel_test;
CREATE DATABASE mlflow;

\connect revintel
CREATE EXTENSION IF NOT EXISTS vector;

\connect revintel_test
CREATE EXTENSION IF NOT EXISTS vector;