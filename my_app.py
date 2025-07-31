import os
from flask import Flask, render_template_string, request
import joblib
import numpy as np
import re
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from tweet_generator import SimpleTweetGenerator

app = Flask(__name__)

#  the pickle file was too big for github , so compressed it
model = joblib.load('like_predictor.pkl')

# Dummy company encoder - radded some samples
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
  <title>T.I.E 🚀🔥</title>
  <!-- Use Playfair Display from Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Playfair+Display:wght@700;900&display=swap" />
  <style>
    :root {
      /* Light mode colors */
      --bg-gradient1: #fd5b63;
      --bg-gradient2: #fca845;
      --bg-gradient3: #fda43a;
      --outer-bg: rgba(255,255,255,0.10);
      --outer-border: #fc477d70;
      --outer-shadow1: #d5845d23;
      --outer-shadow2: #fd5b6387;
      --dot1: #ffffff26;
      --dot2: #fca8454d;
      --dot3: #fd5b635c;
      --container-bg: rgba(255, 255, 255, 0.08);
      --container-text: #231b16;
      --container-shadow1: #652c9080;
      --container-shadow2: #ffbf9d24;
      --container-border-gradient-start: #fc477d;
      --container-border-gradient-mid: #fca845;
      --container-border-gradient-end: #fd5b63;
      --emoji-bar-color: #231b16;
      --emoji-bar-shadow: #fc477d60;
      --heading-color: #231b16;
      --heading-shadow1: #ffec8c60;
      --heading-shadow2: #ffe32540;
      --label-color: #231b16;
      --input-bg: rgba(250, 248, 255, 0.15);
      --input-border: #fda1f5;
      --input-text: #231b16;
      --input-placeholder: #7c6d6daa;
      --focus-border: #fd5b63;
      --focus-bg: rgba(255, 255, 255, 0.95);
      --focus-shadow: #fc477d88;
      --checkbox-label-color: #231b16;
      --checkbox-accent: #fc477d;
      --button-bg-gradient-start: #fc477d;
      --button-bg-gradient-end: #fca845;
      --button-text: #231b16;
      --button-shadow1: #fd5b637c;
      --button-shadow2: #fd5b6331;
      --button-hover-gradient-start: #fca845;
      --button-hover-gradient-end: #fd5b63;
      --button-hover-shadow1: #ff6f794b;
      --button-hover-shadow2: #ff725940;
      --button-active-shadow1: #d04d567d;
      --button-active-shadow2: #d0475267;
      --box-bg: rgba(255, 220, 240, 0.12);
      --box-border: #fc477d66;
      --box-shadow: #ffc78545;
      --selection-bg: rgba(253, 91, 99, 0.26);
      --svg-fill1: #fff9;
      --svg-fill2: #231b16;
      --svg-fill3: #fff;
    }

    .dark-mode {
      /* Dark mode colors */
      --bg-gradient1: #662b2b;
      --bg-gradient2: #6a4a27;
      --bg-gradient3: #9e6b21;
      --outer-bg: rgba(38, 38, 38, 0.6);
      --outer-border: #a8435f70;
      --outer-shadow1: #391c1c70;
      --outer-shadow2: #6f2d3e87;
      --dot1: #ffffff40;
      --dot2: #9470347f;
      --dot3: #a943434c;
      --container-bg: rgba(30, 30, 30, 0.85);
      --container-text: #eee;
      --container-shadow1: #8e427a80;
      --container-shadow2: #cc996333;
      --container-border-gradient-start: #a8435f;
      --container-border-gradient-mid: #c2702e;
      --container-border-gradient-end: #9e6b21;
      --emoji-bar-color: #eee;
      --emoji-bar-shadow: #a8435f60;
      --heading-color: #eee;
      --heading-shadow1: #9e7f1a60;
      --heading-shadow2: #c0a74540;
      --label-color: #eee;
      --input-bg: rgba(60, 60, 60, 0.3);
      --input-border: #c273a1;
      --input-text: #eee;
      --input-placeholder: #bbaaccaa;
      --focus-border: #c2702e;
      --focus-bg: rgba(50, 50, 50, 0.9);
      --focus-shadow: #a8435f88;
      --checkbox-label-color: #eee;
      --checkbox-accent: #bd5b63;
      --button-bg-gradient-start: #a8435f;
      --button-bg-gradient-end: #c2702e;
      --button-text: #eee;
      --button-shadow1: #971e334d;
      --button-shadow2: #7c3e2c31;
      --button-hover-gradient-start: #c2702e;
      --button-hover-gradient-end: #a8435f;
      --button-hover-shadow1: #a8435f6e;
      --button-hover-shadow2: #c2702e6e;
      --button-active-shadow1: #6419287d;
      --button-active-shadow2: #97323267;
      --box-bg: rgba(80, 40, 40, 0.25);
      --box-border: #a8475a66;
      --box-shadow: #a8687745;
      --selection-bg: rgba(180, 110, 110, 0.26);
      --svg-fill1: #eee9;
      --svg-fill2: #eee;
      --svg-fill3: #ddd;
    }

    body {
      margin: 0;
      padding: 0;
      min-height: 100vh;

      /* Use gradients from css variables */
      background:
        repeating-linear-gradient(135deg, rgba(255,255,255,0.07) 0 1px, transparent 1px 40px),
        linear-gradient(135deg, var(--bg-gradient1) 0%, var(--bg-gradient2) 50%, var(--bg-gradient3) 100%);
      font-family: 'Playfair Display', serif;
      display: flex;
      align-items: center;
      justify-content: center;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      position: relative;
      overflow-x: hidden;
      color: var(--container-text);
      transition: background 0.5s ease, color 0.5s ease;
    }


    /* Outer box styling - enclosure for entire UI */
    .outer-box {
      background: var(--outer-bg);
      border: 3.5px solid var(--outer-border);
      border-radius: 34px;
      box-shadow: 0 10px 48px var(--outer-shadow1), 0 2px 8px var(--outer-shadow2);
      padding: 32px 20px;
      margin: 36px 0;
      position: relative;
      max-width: 540px;
      width: 98vw;
      z-index: 2;
      display: flex;
      flex-direction: column;
      align-items: center;
      transition: background 0.5s ease, border-color 0.5s ease, box-shadow 0.5s ease;
    }


    /* Background floating dots */
    .background-dots {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      pointer-events: none;
      z-index: 0;
      background-image:
        radial-gradient(circle at 10% 20%, var(--dot1) 6px, transparent 7px),
        radial-gradient(circle at 80% 10%, var(--dot2) 7px, transparent 8px),
        radial-gradient(circle at 50% 70%, var(--dot3) 9px, transparent 10px);
      background-repeat: no-repeat;
      background-size: cover;
      animation: dots-move 15s linear infinite alternate;
      transition: background 0.5s ease;
    }
    @keyframes dots-move {
      from {
        background-position: 0% 0%, 100% 20%, 50% 80%;
      }
      to {
        background-position: 100% 10%, 90% 100%, 80% 20%;
      }
    }


    /* Container with animated gradient border */
    .container {
      position: relative;
      z-index: 2; /* Above backgrounds */
      max-width: 410px;
      width: 100%;
      padding: 48px 36px 60px;
      background: var(--container-bg);
      border-radius: 2em;
      color: var(--container-text);
      text-align: center;
      font-family: 'Playfair Display', serif;
      box-shadow:
        0 6px 34px 12px var(--container-shadow1),
        0 2px 10px var(--container-shadow2);
      backdrop-filter: blur(8.5px);
      animation: fadein 1.2s cubic-bezier(.17,.67,.83,.67);

      border: 4px solid transparent;
      background-origin: border-box;
      background-clip: padding-box, border-box;
      background-image:
        linear-gradient(to right, var(--container-bg), var(--container-bg)),
        linear-gradient(90deg, var(--container-border-gradient-start), var(--container-border-gradient-mid) 50%, var(--container-border-gradient-end));
      transition: box-shadow 0.3s ease, background 0.5s ease, color 0.5s ease;
    }
    .container:hover {
      box-shadow:
        0 6px 34px 15px #ff7f5080,
        0 3px 15px #ffa35c40;
    }
    @keyframes fadein {
      from {
        opacity: 0;
        transform: translateY(60px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }


    /* Emoji Bar */
    .emoji-bar {
      font-size: 2em;
      margin-bottom: 14px;
      text-align: center;
      letter-spacing: 2px;
      filter: drop-shadow(0 2px 5px var(--emoji-bar-shadow));
      color: var(--emoji-bar-color);
      font-family: 'Playfair Display', serif;
      user-select: none;
      position: relative;
      z-index: 2;
      transition: color 0.5s ease, filter 0.5s ease;
    }


    /* Title */
    h1 {
      color: var(--heading-color);
      text-shadow: 0 4px 16px var(--heading-shadow1), 0 1.5px 3px var(--heading-shadow2);
      font-size: 2.8em;
      letter-spacing: 1.5px;
      margin-bottom: 14px;
      font-weight: 900;
      font-family: 'Playfair Display', serif;
      position: relative;
      z-index: 2;
      transition: color 0.5s ease, text-shadow 0.5s ease;
    }


    /* FORM: vertical layout and full width fields */
    form {
      display: flex;
      flex-direction: column;
      gap: 16px;
      width: 100%;
    }


    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-weight: 700;
      color: var(--label-color);
      font-family: 'Playfair Display', serif;
      font-size: 1.05em;
      transition: color 0.5s ease;
    }


    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      box-sizing: border-box;
      font-size: 1.12em;
      padding: 14px 16px;
      border-radius: 10px;
      border: 1.8px solid var(--input-border);
      background: var(--input-bg);
      color: var(--input-text);
      font-weight: 700;
      font-family: 'Playfair Display', serif;
      outline-offset: 2px;
      outline: none;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
      transition:
        border-color 0.3s ease,
        background-color 0.3s ease,
        box-shadow 0.3s ease,
        color 0.5s ease;
    }


    input::placeholder {
      color: var(--input-placeholder);
      font-style: italic;
      transition: color 0.5s ease;
    }


    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus {
      border-color: var(--focus-border);
      background: var(--focus-bg);
      color: var(--input-text);
      box-shadow: 0 0 8px 2px var(--focus-shadow);
      outline: none;
      transition: border-color 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease, color 0.5s ease;
    }


    /* Checkbox label with horizontal alignment */
    .checkbox-label {
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      color: var(--checkbox-label-color);
      font-family: 'Playfair Display', serif;
      font-size: 1em;
      user-select: none;
      transition: color 0.5s ease;
    }


    input[type="checkbox"] {
      accent-color: var(--checkbox-accent);
      width: 22px;
      height: 22px;
      border-radius: 6px;
      vertical-align: middle;
      margin: 0;
      cursor: pointer;
      transition: accent-color 0.2s ease;
    }
    input[type="checkbox"]:focus {
      outline-offset: 2px;
      outline: 2px solid var(--focus-border);
    }


    /* Button */
    button {
      margin-top: 26px;
      background: linear-gradient(90deg, var(--button-bg-gradient-start), var(--button-bg-gradient-end) 90%);
      border: none;
      border-radius: 12px;
      color: var(--button-text);
      font-weight: 900;
      font-size: 1.32em;
      letter-spacing: 0.6px;
      padding: 16px 0;
      box-shadow: 0 6px 18px var(--button-shadow1), 0 2px 12px var(--button-shadow2);
      cursor: pointer;
      font-family: 'Playfair Display', serif;
      transition: background 0.3s ease, box-shadow 0.3s ease, color 0.3s ease;
      user-select: none;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      position: relative;
      z-index: 2;
    }
    button:hover {
      background: linear-gradient(100deg, var(--button-hover-gradient-start), var(--button-hover-gradient-end));
      box-shadow: 0 8px 22px var(--button-hover-shadow1), 0 3px 15px var(--button-hover-shadow2);
    }
    button:active {
      transform: scale(0.97);
      box-shadow: 0 4px 14px var(--button-active-shadow1), 0 1px 7px var(--button-active-shadow2);
    }


    /* Result, Warning, Error Boxes */
    .result, .warning, .error {
      margin-top: 32px;
      text-align: left;
      font-size: 16px;
      border-radius: 14px;
      padding: 20px 22px 14px;
      word-break: break-word;
      color: var(--container-text);
      border: 2.5px solid var(--box-border);
      background: var(--box-bg);
      box-shadow: 0 0 18px var(--box-shadow);
      font-family: 'Playfair Display', serif;
      user-select: text;
      position: relative;
      z-index: 2;
      transition: background 0.5s ease, border-color 0.5s ease, color 0.5s ease, box-shadow 0.5s ease;
    }
    .result p strong {
      font-size: 1.22em;
    }


    /* Text Selection color */
    ::selection {
      background: var(--selection-bg);
    }

    /* SVG fills in decorative elements */
    svg > circle {
      fill: var(--svg-fill1);
      transition: fill 0.5s ease;
    }
    svg > ellipse {
      fill: var(--svg-fill2);
      transition: fill 0.5s ease;
    }
    svg > rect {
      fill: var(--svg-fill3);
      transition: fill 0.5s ease;
    }

  </style>
</head>
<body>
  <div class="outer-box" role="main">
    <!-- Decorative SVG background overlay -->
    <svg
      style="position:absolute; left:0; top:0; width:100%; height:100%; z-index: 0; pointer-events:none;"
      viewBox="0 0 1200 900" fill="none"
      aria-hidden="true"
    >
      <circle cx="300" cy="400" r="180" opacity="0.04"/>
      <ellipse cx="900" cy="700" rx="260" ry="130" opacity="0.06"/>
      <rect x="900" y="0" width="400" height="200" opacity="0.015" transform="rotate(25 900 0)"/>
    </svg>

    <!-- Decorative floating dots -->
    <div class="background-dots" aria-hidden="true"></div>

    <div class="container" aria-label="Tweet Intelligence Engine Form">
      <div class="emoji-bar" aria-hidden="true">🔥 🚀 👉 👇 😉 🎉 🤔 👏 💡 👍 💰 👀 ✅</div>
      <h1>🐦Tweet Intelligence Engine ✖️<span style="font-size:0.9em">🚀</span></h1>
      <form method="POST" novalidate>
        <label for="company">Company/Brand Name:
          <input
            id="company"
            name="company"
            type="text"
            required
            value="{{ company|default('') }}"
            autocomplete="off"
            placeholder="e.g. Nike 👟"
            aria-required="true"
          />
        </label>
        <label for="tweet_type">Tweet Type:
          <select id="tweet_type" name="tweet_type" aria-required="true">
            {% for tt in supported_tweet_types %}
              <option value="{{ tt }}" {% if tweet_type == tt %}selected{% endif %}>{{ tt.replace('_',' ').title() }}</option>
            {% endfor %}
          </select>
        </label>
        <label for="message">Message or Topic:
          <input
            id="message"
            name="message"
            type="text"
            value="{{ message|default('') }}"
            autocomplete="off"
            placeholder="What’s hot? 🚀🔥"
            aria-required="false"
          />
        </label>
        <label class="checkbox-label" for="has_media">
          <input id="has_media" type="checkbox" name="has_media" {% if has_media %}checked{% endif %} />
          Includes media? <span aria-hidden="true">👀</span>
        </label>
        <label for="hour">Hour to Post (0-23):
          <input
            id="hour"
            name="hour"
            type="number"
            min="0"
            max="23"
            value="{{ hour|default(12) }}"
            aria-required="true"
          />
        </label>
        <button type="submit" class="pulse-btn" aria-label="Predict and Generate Tweet">Predict & Generate 👉</button>

        <button type="button" id="toggle-dark-mode" aria-label="Toggle dark mode">Toggle to Dark Mode</button>
      </form>

      {% if response %}
        {% if response.success %}
          <div class="result" role="region" aria-live="polite" aria-atomic="true">
            <p><strong>Tweet Ready!</strong> 🎉<br />
              <span style="font-size:1.18em;">"{{ response.generated_tweet }}"</span>
            </p>
            <hr />
            <p><strong>Predicted Likes:</strong>
              <span style="font-size:1.25em;">{{ response.predicted_likes }} 👍</span>
            </p>
          </div>
        {% else %}
          <div class="warning" role="alert">⚠️ <strong>Warning:</strong> Tweet generation was not successful.<br>
          Details: {{ response.error or 'Review input and try again.' }}</div>
        {% endif %}
      {% endif %}
      {% if error %}
        <div class="error" role="alert">❌ Error: {{ error }}</div>
      {% endif %}
    </div>
  </div>

  <script>
    // Initialize mode based on localStorage or default to light mode
    const body = document.body;
    const toggleBtn = document.getElementById('toggle-dark-mode');
    const darkModeClass = 'dark-mode';

    function updateToggleButtonText() {
      if(body.classList.contains(darkModeClass)) {
        toggleBtn.textContent = 'Toggle to Light Mode';
      } else {
        toggleBtn.textContent = 'Toggle to Dark Mode';
      }
    }

    toggleBtn.addEventListener('click', () => {
      body.classList.toggle(darkModeClass);
      if(body.classList.contains(darkModeClass)) {
        localStorage.setItem('mode', 'dark');
      } else {
        localStorage.setItem('mode', 'light');
      }
      updateToggleButtonText();
    });

    // On page load, set the mode based on saved preferences
    window.addEventListener('DOMContentLoaded', () => {
      const savedMode = localStorage.getItem('mode');
      if(savedMode === 'dark') {
        body.classList.add(darkModeClass);
      } else {
        body.classList.remove(darkModeClass);
      }
      updateToggleButtonText();
    });
  </script>

  <style>
    /* Pulse button animation */
    .pulse-btn {
      animation: pulse 1.2s infinite;
    }
    @keyframes pulse {
      0% {
        box-shadow: 0 0 0 0 #61dafb33;
      }
      70% {
        box-shadow: 0 0 0 10px transparent;
      }
      100% {
        box-shadow: 0 0 0 0 transparent;
      }
    }
    /* Add small margin to dark mode toggle button */
    #toggle-dark-mode {
      margin-top: 12px;
      background: none;
      border: 2.5px solid var(--button-bg-gradient-start);
      color: var(--button-bg-gradient-start);
      font-family: 'Playfair Display', serif;
      font-weight: 700;
      font-size: 1em;
      border-radius: 10px;
      padding: 8px 16px;
      cursor: pointer;
      transition: background 0.3s ease, color 0.3s ease;
    }
    #toggle-dark-mode:hover {
      background: var(--button-bg-gradient-start);
      color: var(--button-text);
    }
  </style>
</body>
</html>
'''





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
    url_regex = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)' \
                r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+' \
                r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return 1 if re.search(url_regex, str(text)) else 0

@app.route('/', methods=['GET', 'POST'])
def home():
    response = error = None
    company = ''
    tweet_type = 'announcement'
    message = ''
    has_media = False
    hour = 12

    supported_tweet_types = tweet_generator.get_supported_tweet_types()

    if request.method == 'POST':
        company = request.form.get('company', '').strip().lower()
        tweet_type = request.form.get('tweet_type', 'announcement').strip().lower()
        message = request.form.get('message', '').strip()
        has_media = request.form.get('has_media') == 'on'
        try:
            hour = int(request.form.get('hour', 12))
            if not (0 <= hour <= 23):
                hour = 12
        except ValueError:
            hour = 12

        if tweet_type not in supported_tweet_types:
            tweet_type = 'general'

        try:
            # Pass topic only for certain tweet types
            topic = message if tweet_type in ['question', 'tip'] else None

            generated_tweet = tweet_generator.generate_tweet(
                company=company.title(),
                tweet_type=tweet_type,
                message=message if message else None,
                topic=topic if topic else None
            )

            word_count = len(generated_tweet.split())
            char_count = len(generated_tweet)
            polarity = get_polarity(generated_tweet)
            sentiment_str = get_sentiment(polarity)
            sentiment_neg, sentiment_pos = encode_sentiment(sentiment_str)

            # Encode company if known
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
        hour=hour,
        supported_tweet_types=supported_tweet_types
    )



# had to make it global to avoid circular import issues which I was facing  with render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


