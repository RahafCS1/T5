import spacy
from textblob import TextBlob


nlp = spacy.load('en_core_web_sm') # Load spaCy's English modell

# for read from file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

# Load data
tweets = read_file('tweets.txt')

# preprocess tweets
def preprocess_tweet(tweet):
    doc = nlp(tweet)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens).strip()  # Strip leading/trailing whitespace


tweets = [preprocess_tweet(tweet) for tweet in tweets if preprocess_tweet(tweet).strip()]

#  senttiment score
def calculate_sentiment_score(tweet):
    blob = TextBlob(tweet)
    return blob.sentiment.polarity  # Polarity is between -1 (negative) and 1 (positive)

# Calculate sentiment scores for every tweet
tweet_scores = [(tweet, calculate_sentiment_score(tweet)) for tweet in tweets]

# then Sort tweets by sentiment scoore
sorted_tweets = sorted(tweet_scores, key=lambda x: x[1], reverse=True)

# sorted tweets
for tweet, score in sorted_tweets:
    print(f'Score: {score:.2f}, Tweet: {tweet}')
