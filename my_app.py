from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import os

app = Flask(__name__)

# Load your trained StackingRegressor model
model = joblib.load('like_predictor.pkl')

# --- Company encoding (replace/add known companies as necessary for your training data) ---
company_to_code = {
    'example_company': 1,
    'company_a': 2,
    'company_b': 3,
    # Add more companies as needed
}
def encode_company(company_name):
    company_name = company_name.lower().strip()
    return company_to_code.get(company_name, 0)

encoder_sentiment = ['Positive', 'Neutral', 'Negative']

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
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

def has_hashtag(text):
    return 1 if re.search(r'#\w+', str(text)) else 0

def has_url(text):
    url_regex = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)' \
                r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+' \
                r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return 1 if re.search(url_regex, str(text)) else 0

# Placeholder for company average likes statistics (match these to your training scale/mapping)
company_avg_likes_map = {
    'example_company': 123.45,
    'company_a': 67.89,
    'company_b': 45.67,
    # Add more mappings here if needed
}

PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Tweet Generator App</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f6f6f6; }
        .container { padding: 20px; max-width: 400px; margin: 32px auto; background: #fff; border-radius: 12px; }
        input, select { width: 100%; margin: 10px 0; padding: 8px; border-radius: 4px; border: 1px solid #ccc;}
        button { margin-top: 8px; padding: 10px; border: none; color: #fff; background: #007bff; border-radius: 4px; font-size: 16px;}
        .result { border: 2px solid #ccc; border-radius: 8px; padding: 16px; background-color: #f9f9f9; margin-top: 20px; font-family: monospace; }
        .error { color: red; margin-top: 18px; }
        .warning { color: orange; margin-top: 20px; }
        hr { border: 1px dashed #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tweet Generator App</h1>
        <form method="POST">
            <label>Enter a company name:
                <input name="company" type="text" required value="{{ company|default('') }}">
            </label>
            <label>Select tweet type:
                <select name="tweet_type">
                    <option value="announcement" {% if tweet_type=='announcement' %}selected{% endif %}>Announcement</option>
                    <option value="question" {% if tweet_type=='question' %}selected{% endif %}>Question</option>
                    <option value="general" {% if tweet_type=='general' %}selected{% endif %}>General</option>
                    <option value="update" {% if tweet_type=='update' %}selected{% endif %}>Update</option>
                </select>
            </label>
            <label>Enter your message or topic:
                <input name="message" type="text" required value="{{ message|default('') }}">
            </label>
            <button type="submit">Predict & Generate</button>
        </form>

        {% if response %}
            {% if response.success %}
                <div class="result">
                    <p>Tweet Generated Successfully</p>
                    <p>"{{ response.generated_tweet }}"</p>
                    <hr>
                    <p>Predicted Likes: {{ response.predicted_likes }}</p>
                </div>
            {% else %}
                <div class="warning">
                    <p><strong>Warning:</strong> Tweet generation was not successful.</p>
                    <p>Details: {{ response.error or 'Check console for more information.' }}</p>
                </div>
            {% endif %}
        {% endif %}

        {% if error %}
            <p class="error">Error: {{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    company = data.get('company', '').strip().lower()
    tweet_type = data.get('tweet_type', '').strip().lower()
    message = data.get('message', '').strip().lower()

    generated_tweet = f"{tweet_type.capitalize()} about {company}: {message}"

    try:
        company_encoded = encode_company(company)
        content = message
        word_count = len(content.split())
        char_count = len(content)
        has_media = 0
        hour = 12
        sentiment_polarity = TextBlob(content).sentiment.polarity
        sentiment_value = get_sentiment(sentiment_polarity)
        emoji_count = count_emojis(content)
        has_hashtag_val = has_hashtag(content)
        has_url_val = has_url(content)
        tfidf_mean = 0.0
        company_avg_likes = company_avg_likes_map.get(company, 0.0)
        sentiment_neutral = 1 if sentiment_value == 'Neutral' else 0
        sentiment_negative = 1 if sentiment_value == 'Negative' else 0

        features = pd.DataFrame([{
            'word_count': word_count,
            'char_count': char_count,
            'has_media': has_media,
            'hour': hour,
            'company_encoded': company_encoded,
            'emoji_count': emoji_count,
            'has_url': has_url_val,
            'has_hashtag': has_hashtag_val,
            'tfidf_mean': tfidf_mean,
            'company_avg_likes': company_avg_likes,
            'sentiment_Neutral': sentiment_neutral,
            'sentiment_Negative': sentiment_negative
        }])

        predicted_likes = model.predict(features)[0]

        return jsonify({
            'success': True,
            'generated_tweet': generated_tweet,
            'predicted_likes': float(round(predicted_likes, 2))
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/', methods=['GET', 'POST'])
def home():
    response = error = None
    company = tweet_type = message = ''

    if request.method == 'POST':
        company = request.form.get('company', '')
        tweet_type = request.form.get('tweet_type', 'announcement')
        message = request.form.get('message', '')
        try:
            request_data = {
                'company': company,
                'tweet_type': tweet_type,
                'message': message
            }
            with app.test_request_context(json=request_data):
                resp = generate()
                json_data = resp.get_json()
                if json_data.get('success'):
                    response = json_data
                else:
                    error = json_data.get('error', 'Unknown error in prediction')
        except Exception as ex:
            error = str(ex)

    return render_template_string(
        PAGE_TEMPLATE,
        response=response,
        error=error,
        company=company,
        tweet_type=tweet_type,
        message=message
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

