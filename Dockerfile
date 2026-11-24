FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mgts_nav mgts_nav
COPY configs configs
CMD ["python", "-m", "mgts_nav", "--help"]
