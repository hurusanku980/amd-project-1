FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train.py"]
