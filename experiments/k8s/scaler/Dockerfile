FROM python:3.9.25

WORKDIR /app

COPY k8s/scaler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./experiments/

COPY k8s/scaler/*.py ./scaler/

ENV PYTHONPATH=/app/experiments

WORKDIR /app/scaler

VOLUME ["/results"]
VOLUME ["/exp1-results"]

CMD ["python", "main.py"]
