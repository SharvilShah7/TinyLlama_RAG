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
    # Add a more specific prompt
    prompt = f"Answer as a knowledgeable AI: {input_data.text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Add generation parameters to control output
    outputs = model.generate(
        **inputs,
        max_length=200,  # Limit response length
        temperature=0.7,  # Control randomness (lower = more focused)
        top_p=0.9,  # Use nucleus sampling for better diversity
        do_sample=True  # Enable sampling for more natural responses
    )
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}