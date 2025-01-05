import sys
import logging
from scrapy.crawler import CrawlerProcess
from contained_webscrape_processing.webscraper import CrossRefSpider
from contained_webscrape_processing.article_to_json import ArticleProcessor
from pydantic import BaseModel
import traceback

# Define the ClaimQuery class based on the earlier structure
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
    
class ClaimCrawler:
    def __init__(self, claim_query: ClaimQuery):
        print("claim crawler fine")
        self.claim_query = claim_query
        logging.basicConfig(level=logging.INFO)
        self.run_crawl()

    def run_crawl(self):
        claim = self.claim_query.claim
        print(f"Your claim: {claim}")
        try:
            process = CrawlerProcess(settings={
                "USER_AGENT": "Mozilla/5.0",
                "ROBOTSTXT_OBEY": False,
                "DOWNLOAD_TIMEOUT": 10,
                'TELNETCONSOLE_PORT': None,
                "DOWNLOADER_MIDDLEWARES": {
                    'scrapy.downloadermiddlewares.offsite.OffsiteMiddleware': None,
                }
            })

            process.crawl(CrossRefSpider, claim_query = self.claim_query, max_results=100)
            process.start()
        except Exception as e:
            logging.error(f"Error running crawl for claim '{claim}': {e}")
            logging.error(f"Stack trace:\n{traceback.format_exc()}")

        try:
            # Crawl using the claim from the ClaimQuery instance
            print("CRAWLING!")
            process.crawl(CrossRefSpider, claim_query=self.claim_query, max_results=100)
            process.start()
        except Exception as e:
            logging.error(f"Error running crawl for claim '{claim}': {e}")

        # Process the articles after crawling, utilizing the claim
        processor = ArticleProcessor(claim)
        print("ATTEMPTING!")
        processor.process_file("processing_dir/new_articles_content.txt", "processing_dir/retrieved_articles.json")
    
    def print_claim_details(self):
        # Optionally, print details about the claim and its attributes
        print(f"Claim: {self.claim_query.claim}")
        for field, value in self.claim_query.dict().items():
            if field != "claim":  # Avoid printing the claim itself here
                print(f"{field.replace('_', ' ').title()}: {value}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_spider.py <claim>")
        sys.exit(1)

    # Sample ClaimQuery object instantiation (example values)
    claim_query = ClaimQuery(
        claim=sys.argv[1],
        journal_article=True,
        book_chapter=False,
        proceedings_article=True,
        thesis=False,
        report=True,
        patent=False,
        standard=True,
        dataset=False,
        media=True,
        preprint=False,
        dissertation=True
    )

    # Initialize the crawler with the ClaimQuery object
    claim_crawler = ClaimCrawler(claim_query)
    
    # Print claim details (optional)
    claim_crawler.print_claim_details()

    # Run the crawl based on the provided claim query
    try:
        claim_crawler.run_crawl()
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
