FROM python:3.10-slim

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY requirements.txt ./

COPY app.py ./

COPY model_loader.py ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000 4000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
