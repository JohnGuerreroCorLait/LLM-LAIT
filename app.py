from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# URL del modelo
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/doctoradocienciasdelasaludusco.appspot.com/o/LAIT%2Fllama-3.2-1b-instruct-q8_0.gguf?alt=media&token=68958f75-c6e2-4fac-9419-d40576cbffa8"
MODEL_PATH = "llama-3-model.pt"

# Model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer

    # Descargar el modelo si no existe localmente
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model: {response.status_code}")

    # Cargar el modelo y el tokenizador
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("path/to/your-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("path/to/your-model")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model and tokenizer loaded successfully.")

@app.post("/generate")
async def generate_text(request: dict):
    try:
        prompt = request.get("prompt", "")
        max_length = request.get("max_length", 50)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")

        # Generar texto
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
