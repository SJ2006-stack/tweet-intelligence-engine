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
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tweet Generator App - Dark Theme</title>
  <style>
    body, html {
      margin: 0; padding: 0;
      height: 100vh;
      font-family: "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
      background: #18171c;
      /* Subtle grid pattern for depth, like modern dark UIs */
      background-image:
        linear-gradient(90deg, rgba(60,60,70,0.03) 1px, transparent 1px),
        linear-gradient(180deg, rgba(60,60,70,0.03) 1px, transparent 1px);
      background-size: 40px 40px, 40px 40px;
      color: #f4f4fa;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .container {
      background: #232129;
      border-radius: 16px;
      border: 1px solid #35334b;
      width: 100%;
      max-width: 430px;
      padding: 42px 36px 56px;
      box-shadow: 0 8px 32px 4px rgba(20, 13, 47, 0.95), 0 1.5px 3px rgba(84,40,124,0.13);
      text-align: center;
      position: relative;
    }
    h1 {
      font-size: 2.4em;
      margin-bottom: 30px;
      color: #d7a7f9;
      font-family: "Segoe UI Semibold", "Roboto", sans-serif;
      letter-spacing: 1px;
      text-shadow: 0 2px 18px #6125b777;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    label {
      display: flex;
      flex-direction: column;
      font-size: 14.5px;
      font-weight: 500;
      color: #ffeefd;
      text-align: left;
      margin-bottom: 2px;
    }
    input[type="text"],
    input[type="number"],
    select {
      margin-top: 7px;
      padding: 12px 14px;
      font-size: 15px;
      border-radius: 8px;
      border: 1.2px solid #333048;
      background: #222032;
      color: #dacffc;
      outline: none;
      box-shadow: none;
      transition: border-color 0.3s, background 0.2s;
    }
    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus {
      border-color: #a06bf9;
      background: #302943;
      color: #fff;
    }
    .checkbox-label {
      flex-direction: row;
      align-items: center;
      font-weight: 500;
      gap: 11px;
      color: #dacffc;
      text-align: left;
      margin-top: 2px;
    }
    input[type="checkbox"] {
      width: 19px;
      height: 19px;
      accent-color: #e966fa;
      background: #18171c;
      border: 1.5px solid #a06bf9;
      border-radius: 7px;
      transition: accent-color 0.2s;
    }
    button {
      margin-top: 22px;
      background-color: #a06bf9;
      border: none;
      border-radius: 9px;
      color: #fffafd;
      font-weight: 700;
      font-size: 17px;
      letter-spacing: 0.5px;
      padding: 13px 0 11px 0;
      cursor: pointer;
      box-shadow: 0 4px 16px #420a7950, inset 0 -2px 9px #1a0422bb;
      transition: background 0.2s, box-shadow 0.2s;
    }
    button:hover, button:focus {
      background-color: #7d53b5;
      box-shadow: 0 8px 20px #a06bf967, 0 1px 8px #7022fa21;
      outline: none;
    }
    .result, .warning, .error {
      margin-top: 28px;
      text-align: left;
      font-size: 15px;
      border-radius: 8px;
      padding: 18px 18px 12px;
      word-break: break-word;
      color: #f4f4fa;
    }
    .result {
      background: #23202d;
      border: 1.5px solid #7b2ff2;
      font-family: Consolas, monospace;
      box-shadow: 0 0 12px #7b2ff265;
      color: #fbefff;
    }
    .warning {
      background: #322c35;
      border: 1.5px solid #e9b800;
      color: #fffad0;
      box-shadow: 0 0 9px #e9b80055;
    }
    .error {
      background: #2b171b;
      border: 1.5px solid #fd214a;
      color: #ffd3df;
      box-shadow: 0 0 13px #fd214a55;
    }
    hr {
      border: none;
      border-top: 1.5px dashed #574074;
      margin: 15px 0 10px 0;
      opacity: 0.4;
    }
    ::selection {
      background: #a06bf966;
    }
    ::-webkit-scrollbar {
      width: 8px;
    }
    ::-webkit-scrollbar-thumb {
      background: #a06bf9;
      border-radius: 20px;
    }
    ::-webkit-scrollbar-track {
      background: #23202d;
    }
  </style>
</head>
<body>
  <div class="container" role="main">
    <h1>Tweet Generator</h1>
    <form method="POST" novalidate>
      <label for="company">Enter a company name:
        <input id="company" name="company" type="text" required value="{{ company|default('') }}" autocomplete="off" />
      </label>
      <label for="tweet_type">Select tweet type:
        <select id="tweet_type" name="tweet_type" required>
          <option value="announcement" {% if tweet_type=='announcement' %}selected{% endif %}>Announcement</option>
          <option value="question" {% if tweet_type=='question' %}selected{% endif %}>Question</option>
          <option value="general" {% if tweet_type=='general' %}selected{% endif %}>General</option>
          <option value="update" {% if tweet_type=='update' %}selected{% endif %}>Update</option>
        </select>
      </label>
      <label for="message">Enter your message or topic:
        <input id="message" name="message" type="text" required value="{{ message|default('') }}" autocomplete="off" />
      </label>
      <label class="checkbox-label" for="has_media">
        <input id="has_media" type="checkbox" name="has_media" {% if has_media %}checked{% endif %} />
        Has media
      </label>
      <label for="hour">Hour of posting (0-23):
        <input id="hour" name="hour" type="number" min="0" max="23" value="{{ hour|default(12) }}" required />
      </label>
      <button type="submit">Predict & Generate</button>
    </form>
    {% if response %}
      {% if response.success %}
        <div class="result" role="region" aria-live="polite">
          <p><strong>Tweet Generated Successfully</strong></p>
          <p>"{{ response.generated_tweet }}"</p>
          <hr />
          <p><strong>Predicted Likes:</strong> {{ response.predicted_likes }}</p>
        </div>
      {% else %}
        <div class="warning" role="alert">
          <p><strong>Warning:</strong> Tweet generation was not successful.</p>
          <p>Details: {{ response.error or 'Check console for more information.' }}</p>
        </div>
      {% endif %}
    {% endif %}
    {% if error %}
      <p class="error" role="alert">Error: {{ error }}</p>
    {% endif %}
  </div>
</body>
</html>




'''

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

