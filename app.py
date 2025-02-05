from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define request model
class TextInput(BaseModel):
    text: str

# Define API endpoint
@app.post("/analyze")
def analyze(input_data: TextInput):
    input_text = input_data.text

    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Analyze the following sentence for any form of harmful stereotypes, prejudice, or discrimination. This includes racial, gender, sexual orientation, or other forms of bias. Respond in exactly two lines: The first line should be either 'Racist', 'Sexist', 'Homophobic', 'Ableist', or similar categories depending on the content. The second line should explain why the sentence falls into that category, including the negative impact of such discrimination. Sentence: {input_text}"
        }],
        model="llama-3.3-70b-versatile",
    )

    result = chat_completion.choices[0].message.content.strip().split("\n")

    if len(result) < 2:
        raise HTTPException(status_code=400, detail="Invalid LLM response format")

    classification = result[0]
    explanation = result[1]

    return {
        "classification": classification,
        "explanation": explanation
    }

# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI Backend is running!"}
