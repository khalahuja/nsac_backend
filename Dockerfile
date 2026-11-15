# Unified backend with optional heavy astronomy/geospatial packages excluded.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MALLOC_ARENA_MAX=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install -r requirements-runtime.txt
COPY app ./app
COPY data/moon_craters.json ./data/moon_craters.json
RUN useradd --create-home --uid 1000 lunar && chown -R lunar:lunar /app
USER lunar
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/health', headers={'User-Agent':'Selene-Health/1.0'}), timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--no-server-header", "--no-proxy-headers"]
