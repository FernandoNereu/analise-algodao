# Imagem base
FROM python:3.9-slim-bullseye

# Adicionar ferramentas básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar dependências primeiro para aproveitar cache
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Criar um usuário não-root para segurança
RUN useradd -m appuser
USER appuser

# Copiar o código do projeto
COPY . .

# Expor a porta 8501 para Streamlit
EXPOSE 8501

# Comando padrão para rodar o Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.enableCORS=false"]
