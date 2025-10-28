import re
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline as HFBase
import json
import torch
import ast, unicodedata

# ================load and distribute model================
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading model on both GPUs...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} GPU(s).")

if n_gpus > 1:
    # distribute layers automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="balanced"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

model.eval()
print("✅ Model distributed successfully.")

# ================huggingface pipeline================
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

class FixedHuggingFacePipeline(HFBase):
    def predict(self, text, **kwargs):
        return self.pipeline(text)[0]['generated_text']
    apredict = predict
    predict_messages = predict
    apredict_messages = predict

llm = FixedHuggingFacePipeline(pipeline=pipe)

# -langchain components
prompt_template = PromptTemplate(
    template=(
        "You are an expert academic librarian.\n"
        "You are given a raw text block containing a reading list.\n"
        "Your task is to extract each book entry and format it as a JSON list.\n"
        "Each item in the list should be a JSON object with the following keys:\n"
        "- title\n- author\n- year\n- publisher\n"
        "RULES:\n"
        "1. Extract info only from the text.\n"
        "2. If missing, set null.\n"
        "3. Do not invent data.\n"
        "4. Year must be a four-digit number (e.g., 1999, 2015, 2020).\n"
        "5. Author is not publisher.\n"
        "6. Return ONLY a valid JSON array — no markdown, code blocks, or explanations.\n\n"
        "7. If the input data has bracket in it, remove the bracket in the input data.\n"
        "8. Year must be a 4-digit number only. If not found, set year=null.\n"
        "Return ONLY a JSON array — no explanations, no markdown, no other text.\n"
        "Output strictly begins with '[' and ends with ']'.\n"
        "Input Reading List:\n{reading_text}\n\n"
        "JSON Output (only the array):"
    ),
    input_variables=["reading_text"]
)

# langchain parser
parser = JsonOutputParser()

# LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_books(reading_text: str) -> List[Dict[str, Any]]:
    """
    Extract structured book data from a reading list paragraph using a LangChain-managed LLM.
    """
    try:
        raw_output = chain.invoke({"reading_text": reading_text})["text"]
        try:
            return parser.parse(raw_output)
        except Exception:
            # fallback manual clean (similar to your current code)
            cleaned = re.sub(r'```(?:json)?', '', raw_output).replace('`', '')
            cleaned = re.sub(r'("year":\s*)([A-Za-z0-9 ]+)(?=[,\n}])', r'\1"\2"', cleaned)
            cleaned = cleaned.replace("“", '"').replace("”", '"')
            cleaned = unicodedata.normalize("NFKC", cleaned)
            cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8")
            match = re.search(r'\[[\s\S]*\]', cleaned)
            if not match:
                print("No JSON found in output:\n", cleaned)
                return []
            json_text = match.group(0)
            try:
                return json.loads(json_text)
            except Exception:
                json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
                return json.loads(json_text)
                
    except Exception as e:
        print(f"❌❌❌Error during LLM generation or parsing: {e}❌❌❌")
        # print("========== Input text:", reading_text)
        print("========== Output text:", raw_output)
        return []

# ================export excel function================
def export_to_excel(all_reading_lists, filename="course_reading_lists.xlsx"):
    """
    Calls the LLM for each course's reading list and exports results to Excel.
    """
    rows = []

    for course in all_reading_lists:

        raw_list_text = course['reading_list']
        
        if raw_list_text == "Reading list not found.":
            book_data = []
        else:
            print(f"--- Extracting books for {course['course_id']} ---")
            book_data = extract_books(raw_list_text) # Call the LLM here
        
        if not book_data:
             # Add a row even if no books are found, for completeness
             rows.append({
                "Course": course["course_name"],
                "Course ID": course["course_id"],
                "Title": "",
                "Author": "",
                "Year": "",
                "Publisher": ""
             })
        else:
            for b in book_data:
                author = ", ".join(b["author"]) if isinstance(b.get("author"), list) else b.get("author")
                rows.append({
                    "Course": course["course_name"],
                    "Course ID": course["course_id"],
                    "Title": b.get("title"),
                    "Author": author,
                    "Year": b.get("year"),
                    "Publisher": b.get("publisher")
                })

    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"✅ Exported {len(rows)} book entries from {len(all_reading_lists)} courses to {filename}")