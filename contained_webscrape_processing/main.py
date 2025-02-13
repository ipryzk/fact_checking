from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from run_spider import ClaimCrawler

import requests
import json

from classification_runner import ClassificationChecker
from justification_runner import JustificationChecker
app = FastAPI()

# Allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React's local development server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class ClaimQuery(BaseModel):
    claim: str
    journal_article: bool
    book_chapter: bool
    proceedings_article: bool
    report: bool
    standard: bool
    dataset: bool
    posted_content: bool
    dissertation: bool

@app.post("/submit_claim/")
async def submit_claim(claim_query: ClaimQuery):
    
    print("Submitting")

    run = ClaimCrawler(claim_query) # webcrawler
    
    check = ClassificationChecker(
        model_name="microsoft/deberta-large",  # example implementation with classifier
        tokenizer_name="microsoft/deberta-large",     # this ran directly on my host pc so move 
        input_json="processing_dir/retrieved_articles.json"      # implement to cloud if you need remote GPUs
    )
    
    check = JustificationChecker(
        model_name = "ipryzk/model-1b-llama-factcheck-16bit-head",
        tokenizer_name= "ipryzk/model-1b-llama-factcheck-16bit-head",
        input_corroborate = "processing_dir/corroborate.json",
        input_contrast="processing_dir/contradict.json"
    
    )
    
    
    # The URL of the receiving server's /receive-claim/ endpoint
    with open('processing_dir/corroborate_justified.json', 'r') as file:
        corroborate_data = json.load(file)

    with open('processing_dir/contradict_justified.json', 'r') as file:
        contradict_data = json.load(file)

    # Create one single payload combining both
    # Note example I ran within the fact-checking dir for ""Phage therapy can be an effective treatment for antibiotic-resistant bacteria.""
    payload = {
        "corroborate": corroborate_data,
        "contradict": contradict_data
    }
    # Respond with the data back to the React frontend
    return payload
    
    

# TO WORK ON ATTMEPTING TO PROCESS ERRORS, START REACT SCRIPT AND SERVER SIMULTANEOUSLY