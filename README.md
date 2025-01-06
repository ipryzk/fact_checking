Howdy! This is the documentation for Ian Prazak's Webscraper for fact-checking! Within the my-react-app directory, that has the basic implementation for the React component I used to test out the frontend for my project. However, please feel free to skip straight to the webscraping implementation within Python! To test out just the webscraper, go to webcrawltest.py and input a fact-checkable claim. To view the rest of the implementation, check out main.py (that also includes the pipeline for the classification portion of my project). However, the justification was too computationally expensive to run locally, so it was done on a cloud-based system.

Furthermore, the training scripts I used were copied from my older folders contained on my local PC because they contain the models. 

Here is the model pipeline for reference using main.py (run them without the "example" in the name if you wish to test): Inputed claim -> extracted_response_example.json (from CrossRef API) and article_titles_example.txt (purely for programming logging purposes) -> new_articles_content_example.txt - > extracted_response_example.json (the "passes" get filtered out during classification) - > corroborate_example.json and contradict_example.json (the model classified no portions of texts as contrasting) - > sends to justifier via combined corroborate-contradict payload.

Please note, the AI models that WERE trained are not on Github due to storage limitations. They can be found using the following links to Hugging Face: 
https://huggingface.co/ipryzk/llama-finetuned
https://huggingface.co/ipryzk/deberta-large-finetuned

Thanks for stopping by :)