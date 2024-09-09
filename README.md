
# Multilingual text identification with its topic categorization


This project presents an integrated system for automatic language detection 
and topic identification using natural language processing (NLP) models. By 
leveraging the MobileBERT model, known for its efficiency, we achieve 
language detection while minimizing computational resources. The system 
preprocesses data, tokenizes text, and fine-tunes MobileBERT for 
classification tasks. The system incorporates topic identification capabilities 
through models like Latent Dirichlet Allocation (LDA) or BERT-based 
approaches. This addition allows the system not only to detect language but 
also extract meaningful topics from the input text, providing valuable insights. 
Additionally, it also translates the input text to user desired choice. 


## Requirements

Ensure you have Python 3.6+ installed along with the following libraries:

**Python libraries:** warnings, pandas, re, torch,   numpy,  sklearn,  transformers  deep_translator,  langdetect,  ftfy,  os, nltk 

**ServerDeep Learning Framework:** Transformers (Hugging Face Transformers library)  

**Machine Learning Libraries:** scikit-learn (sklearn) 

**Natural Language Processing Libraries:** NLTK   deep_translator, langdetect and ftfy



Install the required packages using:

```bash
pip install -r requirements.txt
```



## Features

1. Develop and integrate a BERT-based language detection model capable of accurately 
identifying multiple languages, including low-resource languages, within a single text 
input. 

2.  Utilize TF-IDF vectorization in conjunction with a fine-tuned BERT classifier to 
enhance the precision and recall of topic categorization in multilingual texts. 

3.  Design the system to effectively manage and categorize text that contains multiple 
languages within the same document, ensuring seamless language transitions and 
accurate topic detection. 

4. Implement advanced text preprocessing steps to handle encoding issues, remove noise, 
and standardize input text, improving the overall quality and consistency of the data fed 
into the models.  

5.  Apply efficient training techniques such as gradient accumulation, mixed precision 
training, and early stopping to reduce computational resource requirements without 
sacrificing model performance.
## Usage

1. Setup Environment:

* Clone the repository.
* Install dependencies using pip install -r requirements.txt.

2. Run the Application:

* Run the main code.

3. Interpret Results

## Contributing 

Contributions are welcome! If you have suggestions, enhancements, or issues, please submit them via GitHub issues.

