# Usa Python 3.12 Slim (Leve e Rápido)
FROM python:3.12-slim

# Variáveis para otimizar o Python no Container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- AQUI ESTÁ O SEGREDO: Instalamos o FFmpeg ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Inicia o Agente
CMD ["python", "main.py"]