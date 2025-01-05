from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contained_webscrape_processing.run_spider import ClaimCrawler
from contained_webscrape_processing.classification_runner import ClassificationChecker
import requests
import json


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
        model_name="ipryzk/deberta-large-finetuned",  # example implementation with classifier
        tokenizer_name="microsoft/deberta-large",     # this ran directly on my host pc so move 
        input_json="replace_with_actual_dataset"      # implement to cloud if you need remote GPUs
    )
     # Prepare the JSON payload to send to the receiver server
    with open('processing_dir/corroborate.json', 'r') as file:
        corroborate_data = json.load(file)

    with open('processing_dir/contradict.json', 'r') as file:
        contradict_data = json.load(file)

    # Create one single payload combining both
    # Note example I ran within the fact-checking dir for ""Phage therapy can be an effective treatment for antibiotic-resistant bacteria.""
    payload = {
        "corroborate": corroborate_data,
        "contradict": contradict_data
    }

    # The URL of the receiving server's /receive-claim/ endpoint
    receiver_url = "http://example/receive-claim" # will need to change based on your cloud ip to integrate justifications

    # Send a POST request to the receiver server with the claim data
    response = requests.post(receiver_url, json=payload)
    

    # Return the response from the receiver server
    return {"status": response.status_code, "response": response.json()}
    

    
 

# TO WORK ON ATTMEPTING TO PROCESS ERRORS, START REACT SCRIPT AND SERVER SIMULTANEOUSLY