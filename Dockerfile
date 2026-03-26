FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY disease_ml ./disease_ml
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]

