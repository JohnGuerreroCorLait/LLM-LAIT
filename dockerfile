FROM python:3.9-slim

# Instalar dependencias
RUN apt-get update && apt-get install -y wget build-essential git

# Clonar llama.cpp y compilar
RUN git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# Copiar archivos de la app
WORKDIR /app
COPY . .

# Instalar dependencias de Python
RUN pip install -r requirements.txt

# Descargar el modelo
RUN wget "https://firebasestorage.googleapis.com/v0/b/doctoradocienciasdelasaludusco.appspot.com/o/LAIT%2Fllama-3.2-1b-instruct-q8_0.gguf?alt=media&token=68958f75-c6e2-4fac-9419-d40576cbffa8" -O llama-3.2-1b-instruct.gguf

# Exponer el puerto y definir el comando de inicio
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
