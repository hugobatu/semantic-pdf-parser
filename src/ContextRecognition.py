import re
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import ast, unicodedata

# Mistral-7B-Instruct-v0.2 model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print("Loading model... (this may take 1–2 minutes on first run)")

# 4-bit quantization to fit on Kaggle T4
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
).eval()

print("✅ Model loaded successfully.")

def extract_books(reading_text: str) -> List[Dict[str, Any]]:
    """
    Extract structured book data from a reading list paragraph using an LLM.
    """

    requirement = """
    You are an expert academic librarian. You are given a raw text block containing a reading list.
    Your task is to extract each book entry and format it as a JSON list.

    Each item in the list should be a JSON object with the following keys:
    - "title": The full title of the book (include edition if present).
    - "author": The author(s) of the book. List all authors if multiple are given.
    - "year": The publication year as a four-digit number if mentioned.
    - "publisher": The publisher if mentioned.

    RULES:
    1. Extract info only from the text.
    2. If missing, set null.
    3. Do not invent data.
    5. Only assign a value to "year" if there is a **four-digit number** that clearly represents a year (e.g., 1999, 2015, 2020). Note that edition is not year.
    5. Return ONLY valid JSON array — no markdown, code blocks, or explanations.
    6. Author is not publisher.
    7. If there is the character "[" or "]" inside the string, remove it because if we keep it, the json format will be broken.
    """
    
    prompt = f"{requirement}\n\nInput Reading List:\n{reading_text}\n\nReturn ONLY the JSON array without any Markdown formatting.\nJSON Output:"

    try:
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # --- CLEAN RESPONSE ---
        cleaned = response
        cleaned = re.sub(r'```(?:json)?', '', cleaned)  # remove fenced code
        cleaned = cleaned.replace('`', '')
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = unicodedata.normalize("NFKC", cleaned)  # normalize weird unicode
        cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8")  # remove hidden chars
        
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            print("No JSON found in model output:\n", cleaned)
            return []

        json_str = match.group(0).strip()

        # --- MULTI-STAGE PARSE ---
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_str)
            except Exception as e:
                # As a final resort, try loose parsing
                json_str = re.sub(r"(\w+):", r'"\1":', json_str)
                json_str = json_str.replace("None", "null")
                print()
                return json.loads(json_str)

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return []