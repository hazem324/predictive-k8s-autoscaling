FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model
COPY model/model.pkl /app/model/model.pkl

# Copy controller files
COPY controller/ ./controller/

# Run application
CMD ["python", "controller/controller.py"]