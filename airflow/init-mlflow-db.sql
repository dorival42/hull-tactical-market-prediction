-- Creates a dedicated mlflow database inside the shared PostgreSQL instance.
-- Runs automatically on first container start (docker-entrypoint-initdb.d/).
CREATE DATABASE mlflow;
