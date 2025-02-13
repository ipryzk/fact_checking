import re
import json
from transformers import GPT2Tokenizer
import sys

class ArticleProcessor:
    def __init__(self, claim, tokenizer_name="gpt2", token_limit=500):
        # Initialize GPT-2 tokenizer and set the token limit
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.token_limit = token_limit
        self.claim = claim  # Store the claim
        self.training = {}

    def split_into_sentences(self, text):
        """
        Split text into sentences using regular expressions,
        handling edge cases like abbreviations.
        """
        # Add space after each punctuation
        text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        return [s.strip() for s in sentences if s.strip()]

    def process_article_content(self, article_content):
        
        sentences = self.split_into_sentences(article_content)
        chunks = []
        current_chunk = []
        current_token_count = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)


            if sentence_token_count > self.token_limit:
                # Split long sentence into smaller parts
                for j in range(0, sentence_token_count, self.token_limit):
                    chunk_tokens = sentence_tokens[j:j + self.token_limit]
                    chunk = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Append chunk only if it ends with complete punctuation
                    if chunk.strip().endswith(('.', '!', '?')):
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                        chunks.append(chunk)
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                        current_chunk.append(chunk)

            else:
                # Handle normal chunking logic
                if current_token_count + sentence_token_count > self.token_limit:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_token_count = sentence_token_count
                else:
                    current_chunk.append(sentence)
                    current_token_count += sentence_token_count

            # Check if the last chunk does not end with a complete sentence
            if current_chunk and not current_chunk[-1].strip().endswith(('.', '!', '?')):
                while i + 1 < len(sentences):
                    next_sentence = sentences[i + 1]
                    next_sentence_tokens = self.tokenizer.encode(next_sentence, add_special_tokens=False)
                    next_sentence_token_count = len(next_sentence_tokens)

                    current_chunk.append(next_sentence)
                    current_token_count += next_sentence_token_count

                    if current_chunk[-1].strip().endswith(('.', '!', '?')):
                        break

                    i += 1  # Move to the next sentence

        # Add the last chunk if any sentences remain and ends with punctuation
        if current_chunk:
            final_chunk = ' '.join(current_chunk)
            if final_chunk.strip().endswith(('.', '!', '?')):
                chunks.append(final_chunk)
            
        return chunks

    def process_file(self, input_file, output_file):
        """
        Process an input file line by line to extract DOIs and article content.
        Writes the processed chunks to the output JSON file as individual JSON objects in a list.
        """
        current_doi = None
        current_type = None
        article_content = ""
        processed_data = []  # List to collect all JSON objects

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract DOI
                doi_match = re.search(r"DOI:\s*([^;\s]+)", line)
                type_match = re.search(r"ARTICLE_TYPE:\s*([^;\s]+)", line)
                if doi_match:
                    if current_doi is not None:
                        # Process the previous DOI's content
                        chunks = self.process_article_content(article_content)
                        for chunk in chunks:
                            processed_data.append({
                                "claim": self.claim,  # Use the stored claim
                                "doi": current_doi,
                                "type": current_type,
                                "text": chunk
                            })
                    # Update DOI for the next article
                    current_doi = doi_match.group(1)
                    current_type = type_match.group(1) if type_match else None
                    article_content = ""  # Reset article content for the new DOI

                # Extract article content
                content_match = re.search(r"ARTICLE CONTENT:\s*(.*)", line, re.DOTALL)
                if content_match:
                    article_content += content_match.group(1).strip() + " "  # Append content

            # Process the last DOI's content
            if current_doi is not None:
                chunks = self.process_article_content(article_content)
                for chunk in chunks:
                    processed_data.append({
                        "claim": self.claim,  # Use the stored claim
                        "doi": current_doi,
                        "type": current_type,
                        "text": chunk
                    })

        # Save the processed data to the output file as a JSON array
        self.save_training_data(output_file, processed_data)

    def save_training_data(self, output_file, data):
        """
        Save the training data into a JSON file as a list of objects.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)