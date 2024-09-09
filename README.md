Objectives:

    1. Develop and integrate a BERT-based language detection model capable of accurately identifying multiple languages, including low-resource languages, within a single text input.
    2.  Utilize TF-IDF vectorization in conjunction with a fine-tuned BERT classifier to enhance the precision and recall of topic categorization in multilingual texts.
    3.  Design the system to effectively manage and categorize text that contains multiple languages within the same document, ensuring seamless language transitions and accurate topic detection.
    4.  Implement advanced text preprocessing steps to handle encoding issues, remove noise, and standardize input text, improving the overall quality and consistency of the data fed into the models.
    5.  Apply efficient training techniques such as gradient accumulation, mixed precision training, and early stopping to reduce computational resource requirements without sacrificing model performance.

Pseudocode for the modules implemented

    1) Preprocessing Module

Function preprocess_text(text):
    text = fix_text_encoding_issues(text)
    text = convert_to_lower_case(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = remove_mentions_and_hashtags(text)
    text = remove_extra_whitespaces(text)
    Return cleaned_text


    2)  Language Detection Module

Function load_language_detection_model():
    If model_directory_exists:
        model = load_model_from_directory(model_directory)
        tokenizer = load_tokenizer_from_directory(model_directory)
    Else:
        model_name = 'bert-base-multilingual-cased'
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name)
        train_language_detection_model(model, tokenizer)
    Return model, tokenizer

Function train_language_detection_model(model, tokenizer):
    train_encodings = tokenize_data(X_train, tokenizer)
    test_encodings = tokenize_data(X_test, tokenizer)
    train_dataset = create_dataset(train_encodings, y_train)
    test_dataset = create_dataset(test_encodings, y_test)
    training_args = define_training_arguments()
    trainer = create_trainer(model, training_args, train_dataset, test_dataset)
    trainer.train()
    save_model_and_tokenizer(model, tokenizer, model_directory)
    evaluate_model(trainer)

Function detect_language(text):
    cleaned_text = preprocess_text(text)
    encoded_text = tokenizer(cleaned_text, return_tensors='pt')
    predictions = model(encoded_text)
    detected_language = get_language_from_predictions(predictions)
    Return detected_language

    3) Topic Detection Module

Function load_topic_detection_model():
    If model_directory_exists:
        model = load_model_from_directory(model_directory)
        tokenizer = load_tokenizer_from_directory(model_directory)
    Else:
        model_name = 'bert-base-uncased'
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name)
        train_topic_detection_model(model, tokenizer)
    Return model, tokenizer

Function train_topic_detection_model(model, tokenizer):
    vectorizer = create_tfidf_vectorizer()
    X_tfidf = vectorize_data(X_topics, vectorizer)
    train_encodings = tokenize_data(X_tfidf_train, tokenizer)
    test_encodings = tokenize_data(X_tfidf_test, tokenizer)
    train_dataset = create_dataset(train_encodings, y_train)
    test_dataset = create_dataset(test_encodings, y_test)
    training_args = define_training_arguments()
    trainer = create_trainer(model, training_args, train_dataset, test_dataset)
    trainer.train()
    save_model_and_tokenizer(model, tokenizer, model_directory)
    evaluate_model(trainer)

Function detect_topic(text):
    cleaned_text = preprocess_text(text)
    encoded_text = tokenizer(cleaned_text, return_tensors='pt')
    predictions = model(encoded_text)
    detected_topic = get_topic_from_predictions(predictions)
    Return detected_topic

    4) Translation Module

Function detect_and_translate_topic(text, target_language):
    sentences = split_text_into_sentences(text)
    detected_topics = []
    detected_languages = []
    translations = []

    For each sentence in sentences:
        detected_lang = detect_language(sentence)
        detected_languages.append(detected_lang)
        
        detected_topic = detect_topic(sentence)
        detected_topics.append(detected_topic)
        
        translated_sentence = translate_text(sentence, target_language)
        translations.append(translated_sentence)
    
    unique_topics = remove_duplicates(detected_topics)
    unique_languages = remove_duplicates(detected_languages)
    
    Return unique_topics, unique_languages, translations

Explanation of Modules

1. Preprocessing Module:
This module is responsible for cleaning the input text by fixing encoding issues, converting to lower case, removing special characters, URLs, mentions, hashtags, and extra whitespaces.
2. Language Detection Module:
Loading the Model: Checks if the language detection model exists locally. If it does, it loads the model and tokenizer; otherwise, it initializes, trains, and saves the model.
Training the Model: Preprocesses the data, tokenizes it, creates datasets, defines training arguments, and trains the model. After training, it evaluates and saves the model.
Detecting Language: Preprocesses the input text, tokenizes it, runs it through the model, and returns the detected language.
3. Topic Detection Module:
Loading the Model: Similar to the language detection module, it checks for an existing model, loads it if available, or trains and saves a new model.
Training the Model: Preprocesses and vectorizes the data using TF-IDF, tokenizes it, creates datasets, defines training arguments, and trains the model. It then evaluates and saves the model.
Detecting Topic: Preprocesses the input text, tokenizes it, runs it through the model, and returns the detected topic.
4. Translation Module:
Translating Text: Uses the GoogleTranslator to translate text from the detected language to the target language.
5. Main Detection and Translation Function:
Processing Input Text: Splits the input text into sentences, detects the language and topic of each sentence, translates each sentence, and returns the unique detected topics, languages, and
Experimental Results

Introduction to the Experimental Setup:

Language Detection Dataset:

The dataset consists of text samples labeled with their respective languages.
The data was cleaned and preprocessed to remove noise such as special characters, numbers, URLs, and unnecessary spaces.

Topic Detection Dataset:

A custom dataset was created with text samples labeled with their respective topics.
The dataset was expanded to include a diverse range of topics such as NLP, History, Politics, Education, Weather, Music and Arts, Engineering, and Artificial Intelligence.
Text samples were cleaned similarly to the language detection dataset to ensure consistency.

Evaluation Metrics

Accuracy:
The proportion of correctly classified samples out of the total samples.

Precision:
The proportion of true positive predictions out of the total positive predictions made by the model.

Recall:
The proportion of true positive predictions out of the total actual positives in the dataset.

F1-Score:
The harmonic mean of precision and recall, providing a single metric that balances both.


