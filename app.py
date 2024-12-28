import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

# Ruta al modelo LLaMA
MODEL_PATH = "llama-3.2-1b-instruct.gguf"

# Verifica que el modelo existe
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model file not found at {MODEL_PATH}")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Construye el comando para ejecutar llama.cpp
        command = [
            "./main",
            "-m", MODEL_PATH,
            "-p", request.prompt,
            "--tokens", str(request.max_tokens)
        ]

        # Ejecuta el comando y captura la salida
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise Exception(f"Error running model: {result.stderr}")

        # Retorna el resultado al cliente
        output = result.stdout
        return {"response": output.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
