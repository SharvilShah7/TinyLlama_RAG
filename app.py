from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

# Define a model for the input data
class TextInput(BaseModel):
    text: str

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/generate")
async def generate(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}