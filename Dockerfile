FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env/         ./env/
COPY static/      ./static/
COPY app.py       .
COPY inference.py .
COPY openenv.yaml .

EXPOSE 7860

RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
