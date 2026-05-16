FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.pkl .
COPY controller.py .
COPY prometheus_client.py .
COPY features.py .
CMD ["python", "controller.py"]