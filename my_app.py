import os
from flask import Flask, render_template_string, request
import joblib
import numpy as np
import re
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from tweet_generator import SimpleTweetGenerator
import random

app = Flask(__name__)

# Load the trained model once at startup
model = joblib.load('like_predictor.pkl')

# Dummy company encoder - replace with your saved encoder or proper logic
company_encoder = LabelEncoder()
company_encoder.fit(['nike', 'starbucks', 'apple', 'tesla', 'our company'])

def encode_sentiment(sentiment_str):
    if sentiment_str == "Positive":
        return (0, 1)
    elif sentiment_str == "Negative":
        return (1, 0)
    else:
        return (0, 0)

company_avg_likes_map = {
    'nike': 150,
    'starbucks': 100,
    'apple': 200,
    'tesla': 180,
    'our company': 120
}

PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<!-- ... HTML/CSS TEMPLATE AS BEFORE ... -->
'''  # CONTINUE your full HTML template code here (truncated for brevity)

# Instantiate the tweet generator once
tweet_generator = SimpleTweetGenerator()

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def count_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

def has_hashtag(text):
    return 1 if re.search(r'#\w+', str(text)) else 0

def has_url(text):
    url_regex = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return 1 if re.search(url_regex, str(text)) else 0

@app.route('/', methods=['GET', 'POST'])
def home():
    response = error = None
    company = ''
    tweet_type = 'announcement'
    message = ''
    has_media = False
    hour = 12

    if request.method == 'POST':
        company = request.form.get('company', '').strip().lower()
        tweet_type = request.form.get('tweet_type', 'announcement')
        message = request.form.get('message', '').strip()
        has_media = request.form.get('has_media') == 'on'
        try:
            hour = int(request.form.get('hour', 12))
            if not (0 <= hour <= 23):
                hour = 12
        except ValueError:
            hour = 12

        try:
            generated_tweet = tweet_generator.generate_tweet(
                company=company.title(),
                tweet_type=tweet_type,
                message=message,
                topic=message
            )
            word_count = len(generated_tweet.split())
            char_count = len(generated_tweet)
            polarity = get_polarity(generated_tweet)
            sentiment_str = get_sentiment(polarity)
            sentiment_neg, sentiment_pos = encode_sentiment(sentiment_str)
            if company in company_encoder.classes_:
                company_encoded = int(company_encoder.transform([company])[0])
            else:
                company_encoded = 0
            emoji_count = count_emojis(generated_tweet)
            url_flag = has_url(generated_tweet)
            hashtag_flag = has_hashtag(generated_tweet)
            tfidf_mean = 0.1
            company_avg_likes = company_avg_likes_map.get(company, 100)
            features = np.array([[
                word_count,
                char_count,
                int(has_media),
                hour,
                company_encoded,
                emoji_count,
                url_flag,
                hashtag_flag,
                tfidf_mean,
                company_avg_likes,
                sentiment_neg,
                sentiment_pos
            ]])
            predicted_likes = model.predict(features)[0]
            predicted_likes = int(round(predicted_likes))
            response = {
                'success': True,
                'generated_tweet': generated_tweet,
                'predicted_likes': predicted_likes,
                'error': None
            }
        except Exception as e:
            error = f"Internal error: {str(e)}"
            response = {'success': False, 'generated_tweet': '', 'predicted_likes': 0, 'error': str(e)}

    return render_template_string(
        PAGE_TEMPLATE,
        response=response,
        error=error,
        company=company,
        tweet_type=tweet_type,
        message=message,
        has_media=has_media,
        hour=hour
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

