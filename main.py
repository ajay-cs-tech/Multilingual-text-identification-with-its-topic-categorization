import warnings
import pandas as pd
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from ftfy import fix_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import nltk
from nltk.tokenize import sent_tokenize
import langid


nltk.download('punkt')
DetectorFactory.seed = 0  # Ensure reproducibility

warnings.filterwarnings("ignore")

# Define max_length for tokenization
max_length = 128

# Load data (replace with your data path)
df = pd.read_csv("Language Detection.csv")

# Data preprocessing
def preprocess_text(text):
    text = fix_text(text)  # Fix text encoding issues
    text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', text.lower())
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('\s+', ' ', text)
    return text

df['Cleaned_Text'] = df['Text'].astype(str).apply(preprocess_text)

# Sample a smaller subset of the data for training to reduce training time
df_sample = df.sample(n=2000, random_state=42)  # Adjust the sample size as needed

# Define features and labels for language detection
X_lang = df_sample['Cleaned_Text']
y_lang = df_sample['Language']
languages = y_lang.unique()

# Encode labels for language detection
label_encoder = {lang: i for i, lang in enumerate(languages)}
inv_label_encoder = {i: lang for lang, i in label_encoder.items()}
y_lang_encoded = y_lang.map(label_encoder)

# Split the data for language detection
X_lang_train, X_lang_test, y_lang_train, y_lang_test = train_test_split(X_lang, y_lang_encoded, test_size=0.2, random_state=42)

# Reset index to ensure alignment
X_lang_train = X_lang_train.reset_index(drop=True)
y_lang_train = y_lang_train.reset_index(drop=True)
X_lang_test = X_lang_test.reset_index(drop=True)
y_lang_test = y_lang_test.reset_index(drop=True)

# Model directory for language detection
model_dir_lang = '/home/ajay/Documents/nlp_nlp/models_nlp'

# Check if language detection model already exists
if not os.path.exists(model_dir_lang):
    # Load pre-trained BERT tokenizer and model for language detection
    model_name_lang = 'bert-base-multilingual-cased'
    tokenizer_lang = BertTokenizer.from_pretrained(model_name_lang)
    model_lang = BertForSequenceClassification.from_pretrained(model_name_lang, num_labels=len(languages))

    # Tokenize data for language detection
    train_encodings_lang = tokenizer_lang(list(X_lang_train), truncation=True, padding=True, max_length=max_length)
    test_encodings_lang = tokenizer_lang(list(X_lang_test), truncation=True, padding=True, max_length=max_length)

    class LanguageDetectionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset_lang = LanguageDetectionDataset(train_encodings_lang, y_lang_train)
    test_dataset_lang = LanguageDetectionDataset(test_encodings_lang, y_lang_test)

    # Function to compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        predictions = pred.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
        }

    # Training arguments for language detection
    training_args_lang = TrainingArguments(
        output_dir='./results_lang',
        num_train_epochs=1,  # Reduced number of epochs
        per_device_train_batch_size=16,  # Increased batch size for faster training
        per_device_eval_batch_size=16,
        warmup_steps=50,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir='./logs_lang',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        fp16=True,
    )

    # Trainer for language detection
    trainer_lang = Trainer(
        model=model_lang,
        args=training_args_lang,
        train_dataset=train_dataset_lang,
        eval_dataset=test_dataset_lang,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # Train the model for language detection
    print("Training the language detection model...")
    trainer_lang.train()

    # Save the trained model and tokenizer
    model_lang.save_pretrained(model_dir_lang)
    tokenizer_lang.save_pretrained(model_dir_lang)

    # Evaluate the model for language detection
    eval_result_lang = trainer_lang.evaluate()
    print("Evaluation results: ", eval_result_lang)
else:
    # Load the trained model and tokenizer for language detection
    model_lang = BertForSequenceClassification.from_pretrained(model_dir_lang)
    tokenizer_lang = BertTokenizer.from_pretrained(model_dir_lang)

# Model directory for topic detection
model_dir_topic = '/home/ajay/Documents/nlp_nlp/models_nlp_topic_8'

# Check if topic detection model already exists
if not os.path.exists(model_dir_topic):
    # Create the TF-IDF vectorizer for topic detection
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    # Sample data for topic detection
    data = [
        {"text": "Large language models are transforming NLP.", "topic": "NLP"},
        {"text": "The fall of the Berlin Wall marked the end of the Cold War.", "topic": "History"},
        {"text": "Political campaigns aim to win voter support.", "topic": "Politics"},
        {"text": "Special education caters to the needs of students with disabilities.", "topic": "Education"},
        {"text": "Snowstorms can disrupt transportation systems.", "topic": "Weather"},
        {"text": "Surrealism explored the unconscious mind through art.", "topic": "Music and Arts"},
        {"text": "Computer engineering integrates computer science and electronic engineering.", "topic": "Engineering"},
        {"text": "AI algorithms can learn from data to make predictions.", "topic": "Artificial Intelligence"},
        {"text": "Language modeling is crucial for understanding context in NLP.", "topic": "NLP"},
        {"text": "The history of the Roman Empire spans over a millennium.", "topic": "History"},
        {"text": "Diplomacy is essential for maintaining international relations.", "topic": "Politics"},
        {"text": "STEM education includes science, technology, engineering, and mathematics.", "topic": "Education"},
        {"text": "Monsoons bring heavy rainfall to certain regions.", "topic": "Weather"},
        # Add more examples...
    ]

    # Expand the dataset to include 1000 more examples
    data.extend([
        {"text": f"This is example text {i} for topic detection.", "topic": np.random.choice(["Technology", "Artificial Intelligence", "History", "Politics", "Education", "Weather", "Music and Arts", "Engineering"])}
        for i in range(1)
    ])

    df_topics = pd.DataFrame(data)
    df_topics['Cleaned_Text'] = df_topics['text'].apply(preprocess_text)

    X_topics = df_topics['Cleaned_Text']
    y_topics = df_topics['topic']

    # Vectorize the topic data
    X_tfidf = vectorizer.fit_transform(X_topics)
    topic_labels = y_topics.unique()

    # Encode labels for topic detection
    label_encoder_topics = {topic: i for i, topic in enumerate(topic_labels)}
    inv_label_encoder_topics = {i: topic for topic, i in label_encoder_topics.items()}
    y_topics_encoded = y_topics.map(label_encoder_topics)

    # Split the data for topic detection
    X_topics_train, X_topics_test, y_topics_train, y_topics_test = train_test_split(X_tfidf, y_topics_encoded, test_size=0.2, random_state=42)

    # Reset index to ensure alignment
    X_topics_train = pd.DataFrame(X_topics_train.toarray()).reset_index(drop=True)
    X_topics_test = pd.DataFrame(X_topics_test.toarray()).reset_index(drop=True)
    y_topics_train = y_topics_train.reset_index(drop=True)
    y_topics_test = y_topics_test.reset_index(drop=True)

    # Convert NumPy array to list of strings for tokenizer
    X_topics_train_list = [" ".join([str(num) for num in row]) for row in X_topics_train.to_numpy()]
    X_topics_test_list = [" ".join([str(num) for num in row]) for row in X_topics_test.to_numpy()]

    # Load pre-trained BERT tokenizer and model for topic detection
    model_name_topic = 'bert-base-uncased'
    tokenizer_topic = BertTokenizer.from_pretrained(model_name_topic)
    model_topic = BertForSequenceClassification.from_pretrained(model_name_topic, num_labels=len(topic_labels))

    # Tokenize data for topic detection
    train_encodings_topic = tokenizer_topic(X_topics_train_list, truncation=True, padding=True, max_length=max_length)
    test_encodings_topic = tokenizer_topic(X_topics_test_list, truncation=True, padding=True, max_length=max_length)

    class TopicDetectionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset_topic = TopicDetectionDataset(train_encodings_topic, y_topics_train)
    test_dataset_topic = TopicDetectionDataset(test_encodings_topic, y_topics_test)

    def compute_metrics_topic(pred):
        labels = pred.label_ids
        predictions = pred.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
        }

    # Trainer for topic detection
    training_args_topic = TrainingArguments(
        output_dir='./results_topic',
        num_train_epochs=1,  # Reduced number of epochs
        per_device_train_batch_size=16,  # Increased batch size for faster training
        per_device_eval_batch_size=16,
        warmup_steps=50,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir='./logs_topic',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        fp16=True,
    )

    trainer_topic = Trainer(
        model=model_topic,
        args=training_args_topic,
        train_dataset=train_dataset_topic,
        eval_dataset=test_dataset_topic,
        compute_metrics=compute_metrics_topic,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # Train the model for topic detection
    print("Training the topic detection model...")
    trainer_topic.train()

    # Save the trained model and tokenizer
    model_topic.save_pretrained(model_dir_topic)
    tokenizer_topic.save_pretrained(model_dir_topic)
else:
    # Load the trained model and tokenizer for topic detection
    model_topic = BertForSequenceClassification.from_pretrained(model_dir_topic)
    tokenizer_topic = BertTokenizer.from_pretrained(model_dir_topic)

# Function to detect transliterated Hindi and Kannada
import langid

def detect_transliterated_lang(text):
    # Use langid to detect the language
    lang, _ = langid.classify(text)
    
    # Additional rules to detect Hindi, Kannada, Tamil, and Telugu transliterated text
    hindi_keywords = [
        "hai", "ho", "hu", "hun", "tha", "thi", "hoon", "kar", "karna", "karta", "karti", "karo", "kare",
        "ke", "ko", "se", "aur", "or", "par", "lekin", "magar", "ya",
        "namaste", "prasad", "khana", "pani", "ghar", "school", "gaadi", "dost", "dosto", "bhai", "behen",
        "kya", "kaise", "kahan", "kab", "kaun", "kitna", "kyun"
    ]
    
    kannada_keywords = [
        "naanu", "neenu", "avalu", "ivaru", "idhe", "illa", "madi", "maadi", "beda", "helri", "helthini", "kelri", "kelthini",
        "alli", "illi", "matte", "alva", "athava", "henge", "andu", "aagi",
        "namaskara", "anna", "akshara", "shala", "mane", "benki", "neeru", "channagide", "kelasa", "magu", "tande", "thayi",
        "yenu", "yaaru", "yaake", "hege", "yavaga", "yelli", "eshtu"
    ]

    tamil_keywords = [
        "naan", "nee", "avan", "avanga", "vandhu", "irukku", "illai", "seyyum", "kudunga", "kuduthen", "padikkum",
        "enna", "yen", "inge", "ange", "saar", "seri", "enakku", "unakku", "namakku",
        "vanakkam", "saapadu", "thanni", "veedu", "paatu", "thambi", "akka", "thozhi",
        "enna", "epdi", "enga", "evlo", "yaar", "yen", "eppadi", "enna"
    ]

    telugu_keywords = [
        "nenu", "neevu", "aame", "vaadu", "vastadu", "undi", "ledu", "chesthunna", "ivvandi", "icchanu", "chadivthundi",
        "em", "enduku", "ikada", "akada", "annayya", "akka", "mari", "nakku", "mikku", "manaku",
        "namaskaram", "bhojanam", "neellu", "illu", "paata", "tammudu", "akka", "sneham",
        "em", "ela", "ekkada", "enta", "evaru", "enduku", "eppudu", "emi"
    ]

    if any(word in text.lower() for word in hindi_keywords):
        return 'hi'
    elif any(word in text.lower() for word in kannada_keywords):
        return 'kn'
    elif any(word in text.lower() for word in tamil_keywords):
        return 'ta'
    elif any(word in text.lower() for word in telugu_keywords):
        return 'te'
    else:
        return lang


# Function to detect and translate support ticket
def detect_and_translate_topic(text, target_lang):
    # Split the text into sentences for better handling
    sentences = sent_tokenize(text)
    
    detected_topics = []
    detected_languages = []
    translations = []

    for sentence in sentences:
        # Detect language
        detected_lang = detect_transliterated_lang(sentence)
        detected_languages.append(detected_lang)
        
        # Tokenize sentence for topic detection
        encoded_sentence = tokenizer_topic(sentence, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        
        # Predict topic
        with torch.no_grad():
            output = model_topic(**encoded_sentence)
            predicted_topic_idx = torch.argmax(output.logits, dim=1).item()
            predicted_topic = inv_label_encoder_topics[predicted_topic_idx]
            detected_topics.append(predicted_topic)
        
        # Translate sentence
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_sentence = translator.translate(sentence)
        translations.append(translated_sentence)
    
    # Aggregate detected topics and languages without duplicates
    unique_topics = list(set(detected_topics))
    unique_languages = list(set(detected_languages))
    
    return unique_topics, unique_languages, translations

input_text = input("Enter the text you want to identify and translate: ")
target_language = input("Enter the target language for translation (e.g., 'en' for English, 'fr' for French, 'es'): ")

print("Running detection and translation...")
topics, detected_languages, translation = detect_and_translate_topic(input_text, target_language)

#print(f"The detected topics are: {topics}")
print(f"The detected languages are: {detected_languages}")
#print(f"Translation to {target_language}: {translation}")


