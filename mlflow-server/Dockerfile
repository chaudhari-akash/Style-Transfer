FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN pip install mlflow==2.8.1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install Pillow==10.0.0
RUN pip install requests

COPY custom_model_wrapper.py /app/
COPY import_custom_model.py /app/
COPY model.py /app/

ENTRYPOINT ["python", "/app/import_custom_model.py"]
