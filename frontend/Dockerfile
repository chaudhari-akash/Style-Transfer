FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

COPY app.py ./

COPY templates/ ./templates/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app", "--timeout", "240"] 