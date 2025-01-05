Hi! This is the documentation for Ian Prazak's Webscraper for fact-checking!
Within the my-react-app directory, that has the basic implementation for the React component I used to test out the frontend for my project.
However, please feel free to skip straight to the webscraping implementation within Python! To test out just the webscraper, go to webcrawltest.py
and input a fact-checkable claim. To view the rest of the implementation, check out main.py (that also includes the pipeline for the classification portion
of my project). However, the justification was too computationally expensive to run locally, so it was done on a cloud-based system.

Here is the model pipeline for reference using main.py (run them without the "example" in the name if you wish to test):

Inputed claim - > extracted_response_example.json (from CrossRef API)  - > article_titles_example.txt (purely for programming logging purposes)
                    |
                    v   
    new_articles_content_example.txt - > extracted_response_example.json (the "passes" get filtered out during classification)
                                        |                           |
                                        v                           v
                        corroborate_example.json    contradict_example.json (the model classified no portions of texts as contrasting)
                                        |                           |
                                        v                           v
                                        sends to justifier via combined corroborate-contradict payload
                                        