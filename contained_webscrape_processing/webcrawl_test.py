from typing import Union
from pydantic import BaseModel
from run_spider import ClaimCrawler

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


claim_test = ClaimQuery(
    claim = "Phage therapy can be an effective treatment for antibiotic-resistant bacteria.",
    journal_article=True,
    book_chapter = True,
    proceedings_article = True,
    report = True,
    standard = True,
    dataset = True, 
    posted_content = True,
    dissertation = True
)
def submit_claim(claim_query: ClaimQuery):
    print("Submitting; outbound")
    run = ClaimCrawler(claim_query)

submit_claim(claim_test)