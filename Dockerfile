FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir matplotlib==3.8.2 jupyter

COPY . .