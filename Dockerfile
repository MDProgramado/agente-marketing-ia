# Usa uma imagem Python 3.12 leve e segura
FROM python:3.12-slim

# Define variáveis de ambiente para o Python não criar arquivos .pyc e logs aparecerem na hora
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cria a pasta de trabalho dentro do container
WORKDIR /app

# Instala ferramentas básicas do sistema (necessário para algumas libs)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o seu código para dentro do container
COPY . .

# Comando para iniciar o agente
CMD ["python", "main.py"]