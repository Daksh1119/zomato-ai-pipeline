# Airflow 3.x image + DuckDB + FAB auth (for admin/admin login).
FROM apache/airflow:3.0.3-python3.11

# DuckDB is a self-contained wheel. Airflow's constraints file omits it, so
# install it without dependency resolution after the constrained provider.
# A locally cached Linux wheel is optional (and git-ignored) for slow or
# offline Docker networks; fresh clones use PyPI as usual.
COPY vendor/ /tmp/duckdb-wheel/
RUN pip install --no-cache-dir apache-airflow-providers-fab && \
    if find /tmp/duckdb-wheel -name '*.whl' -print -quit | grep -q .; then \
        PIP_CONSTRAINT= pip install --no-deps /tmp/duckdb-wheel/*.whl; \
    else \
        PIP_CONSTRAINT= pip install --no-deps --retries 5 --resume-retries 5 --timeout 120 "duckdb==1.1.3"; \
    fi
