from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Carga del modelo
model_name = "Meta/LLaMA-3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class Query(BaseModel):
    prompt: str

@app.post("/chat")
def chat(query: Query):
    inputs = tokenizer(query.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
