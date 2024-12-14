import nltk
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
from fuzzywuzzy import process


# Ensure necessary NLTK dependencies are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Combined list of unique article URLs
article_urls = []

# Initialize an empty corpus
corpus = ""

# Download and parse each article, appending the text to the corpus
for url in article_urls:
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        corpus += article.text  # Concatenate the text from each article
    except Exception as e:
        print(f"Error processing {url}: {e}")  # Log errors

# Tokenization into sentences
sentence_list = nltk.sent_tokenize(corpus)

def get_corpus():
    return corpus

def get_sentence_list():
    return nltk.sent_tokenize(corpus)


# Load common questions from JSON file
def load_common_questions(filename='chatbot_data.json'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data.get("common_questions", {})
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file structure.")
    return {}

common_questions = load_common_questions()

# List of diseases for keyword matching
diseases = ['malaria', 'hiv', 'cholera', 'cancer', 'tuberculosis', 'pneumonia']

# Function to get the chatbot greeting response
def greeting_response(text):
    text = text.lower()
    bot_greetings = [
        'hello', 'hi', 'hey',
        'Hello, do you have anything to ask related to disease?', 
        'howdy', 
        'Hi, how can I help you?', 
        'Hello, do you have any queries?', 
        'Hi, welcome to Dr. Bot!', 
        'Yes, ask me about diseases!',
        'Hi, I’m good!', 
        'Hey, great!',
    ]
    user_greetings = [
        'hi', 'hello doctor bot', 'hello', 
        'morning', 'afternoon', 'greetings', 
        'hey', 'how are you', 'can I ask you?', 
        'do you have answers', 'wassup', 
        'whatsap', 'whatsup', 'good morning', 
        'good afternoon', 'good evening',
        'Hi.'
    ]
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)
    return None

# Farewell keywords and responses
farewell_keywords = {'bye', 'okey', 'ok', 'welcome', 'thanks', 'thank you', 'quit', 'see you later', 'chat you later', 'next time', 'great', 'see you', 'it worked'}
farewell_responses = [
    "Chat you later!",
    "Thank you for visiting!",
    "You are welcome!",
    "Next time feel free to ask!",
    "I'm here for you!",
    "I'm glad that I could assist you!",
    "It's great to chat with you!",
    "Continue seeking help!"
]

def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))
    for i in range(length):
        for j in range(length):
            if list_var[list_index[i]] > list_var[list_index[j]]:
                # Swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
                
    return list_index

# Function to get the bot response
def bot_response(user_input, sentence_list):
    user_input = user_input.lower()

    # Check for farewell keywords
    if user_input in farewell_keywords:
        return random.choice(farewell_responses)

    # Check for predefined responses first using fuzzy matching
    best_match = process.extractOne(user_input, common_questions.keys())
    if best_match and best_match[1] >= 70:  # Match threshold of 70%
        return common_questions[best_match[0]]

    # Tokenization and keyword matching for other diseases
    sentence_list.append(user_input)

    # Use TF-IDF Vectorizer for better context understanding
    tfidf_vectorizer = TfidfVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(tfidf_vectorizer[-1], tfidf_vectorizer)
    similarity_scores_list = similarity_scores.flatten()

    # Get indices of sentences based on similarity scores
    index = index_sort(similarity_scores_list)
    index = index[1:]  # Remove the user's input

    bot_response = ''
    j = 0

    # Improved logic to match context better
    for i in range(len(index)):
        if j >= 3:  # Adjust this limit as needed
            break
        if index[i] < len(sentence_list) and similarity_scores_list[index[i]] > 0.1:
            # Use a loop to check if any disease keyword matches
            for disease in diseases:
                if disease in user_input and disease in sentence_list[index[i]].lower():
                    bot_response += ' ' + sentence_list[index[i]]
                    j += 1
                    break  # Break after finding the first matching disease

    # If no relevant match is found, provide a fallback response
    if bot_response == '':
        bot_response = "I couldn't find a direct answer in the articles related to your query. Please try to ask basic questions that I should be able to process without any difficulties."

    sentence_list.remove(user_input)  # Remove user input from the list
    return bot_response
