from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(filename='api_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Define a model for the input data
class TextInput(BaseModel):
    text: str

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/generate")
async def generate(input_data: TextInput):
    # Log the incoming request
    logging.info(f"Received request: {input_data.text}")
    # Provide context to guide the model
    context = "You are an AI assistant specializing in natural language processing. TinyLlama is a compact language model developed by the xAI team, designed for efficient text generation on resource-constrained devices."
    prompt = f"{context}\n\nQuestion: {input_data.text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generation parameters
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Log the response
    logging.info(f"Generated response: {response}")
    return {"response": response}