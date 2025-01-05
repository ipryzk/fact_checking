import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import re
import time
import sys
from pydantic import BaseModel

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


class CrossRefSpider(scrapy.Spider):
    name = 'crossref'
    allowed_domains = ['doi.org']

    def __init__(self, claim_query, max_results=100, *args, **kwargs):
        super(CrossRefSpider, self).__init__(*args, **kwargs)
        self.claim_query = claim_query
        self.max_results = max_results
        self.json_data = None
        self.all_articles_text = []  # List to store the text from all articles
        self.urls = []
        self.start_time = time.time()
        self.article_title = None
        self.times = {
            "start_time": self.start_time,
            "start_requests_time": None,
            "parse_articles_time": None,
            "storejson_time": None,
            "parse_article_content_time": None,
            "extract_div_texts_time": None,
            "clean_text_time": None,
            "closed_time": None
        }


    def construct_url(self):
        # Start with the base URL
        base_url = "https://api.crossref.org/works?"
        
        # Start with the claim query
        url_params = f"query={self.claim_query.claim}&rows={self.max_results}"

        # Filter based on boolean values from ClaimQuery
        filters = []
        if self.claim_query.journal_article:
            filters.append("type:journal-article")
        if self.claim_query.book_chapter:
            filters.append("type:book-chapter")
        if self.claim_query.proceedings_article:
            filters.append("type:proceedings-article")
        if self.claim_query.posted_content:
            filters.append("type:posted-content")
        if self.claim_query.report:
            filters.append("type:report")
        if self.claim_query.standard:
            filters.append("type:standard")
        if self.claim_query.dataset:
            filters.append("type:dataset")
        if self.claim_query.dissertation:
            filters.append("type:dissertation")

        # Add filters to the URL if there are any
        if filters:
            url_params += "&filter=" + ",".join(filters)

        # Construct the full URL
        full_url = base_url + url_params
        return full_url
    
    def start_requests(self):
        self.times["start_requests_time"] = time.time()
        # OLD, NEW ONE PLAYING W/O PRE PRINT https://api.crossref.org/works?query={self.query}&rows={self.max_results} 

        articles_url = self.construct_url()
        self.logger.info(f"Fetching articles from: {articles_url}")
        yield scrapy.Request(url=articles_url, callback=self.parse_articles)

    def parse_articles(self, response):
        if response.status == 200:
            self.json_data = response.json()
            # Save the JSON response to a file for reference and record time
            storejson_time = time.time()
            self.times["storejson_time"] = storejson_time
            with open("processing_dir/extracted_response.json", "w", encoding="utf-8") as file:
                file.write(response.text)

            items = self.json_data.get('message', {}).get('items', [])
            
            # Extract DOI links from the search results

            for item in items:
                doi_id = item.get('DOI')
                article_type = item.get('type')
                self.article_title = item.get('title')[0]
                with open("processing_dir/article_titles.txt", "a", encoding="utf-8") as file:
                    file.write(self.article_title + "\n")
                
                if doi_id:
                    url = f"https://doi.org/{doi_id}"
                    self.urls.append(url)
                    # APPEND PUBLICATIO TYPE
                    # Send the doi to the parse_article_content method
                    yield scrapy.Request(url, callback=self.parse_article_content, meta={'doi_id': doi_id ,'article_type': article_type})
                    # META INCLUDES 'article_type': article_type
        
        self.times["parse_articles_time"] = time.time()
        # Make input that grabs the first few article titles, prints to terminal, and asks for confirmation for relevancy to continue

        self.logger.info(f"Time taken in parse_articles: {self.times['parse_articles_time'] - self.times['start_requests_time']} seconds")

    def parse_article_content(self, response):
        try:
            content_type = response.headers.get('Content-Type', b'').decode('utf-8')
            if response.status == 200 and content_type.startswith('text/html'): # check if div is present in the response:
                # Record time immediately before processing content
                content_start_time = time.time()
                soup = BeautifulSoup(response.text, 'html.parser')
                div_elements = self.extract_div_texts(soup)

                if not div_elements:
                    return

                # Find the <div> with the maximum word count
                max_div = max(div_elements, key=lambda d: len(d[1].split()), default=None)

                if max_div:
                    # Append the content to the list if its greater than 2000 words. Also, append the DOI URL in the beginning
                    if len(max_div[1].split()) > 1250:
                        doi_id = response.meta.get('doi_id')
                        article_type = response.meta.get('article_type')
                        self.all_articles_text.append(f"DOI: {doi_id}; ARTICLE_TYPE:{article_type}; DIV LENGTH: {len(max_div[1].split())}; ARTICLE CONTENT: {max_div[1]}")
                        # ARTICLE_TYPE: {article_type}
                self.times["parse_article_content_time"] = time.time() - content_start_time
                self.logger.info(f"Time taken in parse_article_content: {self.times['parse_article_content_time']} seconds")
            else:
                self.logger.warning(f"Skipping response due to content type: {content_type}")
                return
        except Exception as e:
            self.logger.error(f"Failed to parse the response: {e}")
            self.logger.error(f"Response headers: {response.headers}")
            self.logger.error(f"Response body snippet: {response.text[:500]}")
            return


    def extract_div_texts(self, soup):
        extract_start_time = time.time()
        self.times["extract_div_texts_time"] = extract_start_time

        body = soup.find('body')
        div_elements = []

        if body:
            for div in body.find_all('div', recursive=True):
                if self.is_relevant_div(div):
                    text_start_time = time.time()
                    text = self.clean_text(div.get_text())
                    self.times["clean_text_time"] = time.time() - text_start_time
                    div_elements.append((div, text))
        
        self.times["extract_div_texts_time"] = time.time() - extract_start_time
        return div_elements

    def is_relevant_div(self, div):
        # Check if the <div> has a significant amount of content
        return len(div.get_text(strip=True)) > 100

    def clean_text(self, text):
  
    
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Define the regex pattern to match reference sections
        pattern_references = re.compile(
            r'(?:References|Acknowledgments|Bibliography|Appendix|Google Scholar|PubMed/NCBI|Article|Chapter|MathSciNet|pp\.\s?\d{1,4}|Citations)[^\n]*?'
            r'[\s\S]*?(?=\d{1,4}\.|$)',  # Lookahead for the next reference number or end of string
            flags=re.IGNORECASE
        )

        # Compile the second pattern for volume and issue numbers
        pattern_volume_issue = re.compile(
            r'Volume\s*(\d+)(?:\s*Issue\s*(\d+))?',
            flags=re.IGNORECASE
        )
        
        text = re.sub(pattern_references, '', text)
        text = re.sub(pattern_volume_issue, '', text)
        
        threeormore_fullnames = r"(?:(?:[A-Z][a-z]*(?:\s[A-Z][a-z]+)*)\s*(?:,\s*)?){6,}(?:et\.?\s?al\.?)?"

        text = re.sub(threeormore_fullnames, '', text)

        initials_pattern = r"((?:[A-Z]\.\s?\-?[A-Z]\.\s?\-?|[A-Z][a-z]*)(?:,\s*)?){6,}(?:et\.?\s?al\.?)?"

        text = re.sub(initials_pattern, '', text)

        remove_empty_punc = r"\s+[!;,./?]"
        
        link_pattern = r"(?:https?://\S+|doi\.org/\S+)"

        text = re.sub(link_pattern, '', text)

        and_pattern = r'([A-Z][a-z]+ [A-Z][a-z]* (?:and [A-Z][a-z]+ [A-Z][a-z]*)?)'

        # Find all matches
        matches = re.findall(and_pattern, text)

        # Check if the matched pattern occurs three times
        if len(matches) >= 3:
            text = re.sub(and_pattern, '', text)
            text = re.sub(r'\s*\.\s*', '.', text)
            
        pattern = r'(\b\d*\s*(?:Journal|Text|Book|Press)\s+of\s+\w+(?:\s+\w+)*\s*\d*\b|\b\d*\s*\w+(?:\s+\w+)*\s*(?:Journal|Text|Book|Press)\b)'

        matches = re.findall(pattern, text)

        #Check if the matched pattern occurs three times
        if len(matches) >= 3:
            text = re.sub(pattern, '', text)
            text = re.sub(r'(?<=[^\s])[\.,;!?]+|[.,;!?]+(?=\s)' '.', text)
            print(text)

        # Return cleaned text with extra spaces stripped
        text = re.sub(remove_empty_punc, '', text)
    
        return self.remove_text_b4_article_title(text).strip()

    def remove_text_b4_article_title(self, text):
        if self.article_title:
            trace = re.search(re.escape(self.article_title), text, re.IGNORECASE)
        
            if trace:
                # Return the text starting from the title
                return text[trace.start():]
            else:
                # If the title is not found, return the original text
                return text
        else:
            # If no article title is set, return the original text
            return text
    
    def closed(self, reason):
        closed_start_time = time.time()
        self.times["closed_time"] = closed_start_time

        # Save or handle the aggregated results when the spider closes
        with open("processing_dir/new_articles_content.txt", "w", encoding="utf-8") as file:
            for article_text in self.all_articles_text:
                file.write(article_text + "\n\n")  # Separate articles by a new line
        print(f"Saved content from {len(self.all_articles_text)} articles to new_articles_content.txt")
        # Also print total word count
        total_word_count = sum(len(article_text.split()) for article_text in self.all_articles_text)
        print(f"Total word count: {total_word_count}")
        print(self.urls)
        
        # Print the duration
        duration = time.time() - self.start_time
        print(f"Execution time: {duration} seconds")
        self.logger.info(f"Time taken in closed: {self.times['closed_time'] - self.start_time} seconds")

        # Print timing details for each component
        for key, value in self.times.items():
            if value is not None:
                print(f"{key}: {value} seconds")

