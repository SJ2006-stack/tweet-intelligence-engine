from flask import Flask, render_template_string, request
import requests

app = Flask(__name__)

# HTML template (can be split into a separate file if desired)
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

@app.route('/', methods=['GET', 'POST'])
def home():
    response = error = None
    company = tweet_type = message = ''
    if request.method == 'POST':
        company = request.form.get('company', '')
        tweet_type = request.form.get('tweet_type', 'announcement')
        message = request.form.get('message', '')
        try:
            # Call the Tweet Generator backend (week3_api.py)
            res = requests.post(
                'http://127.0.0.1:5001/generate',
                headers={'Content-Type': 'application/json'},
                json={
                    'company': company,
                    'tweet_type': tweet_type,
                    'message': message
                },
                timeout=10
            )
            if res.ok:
                response = res.json()
            else:
                error = f'API request failed: {res.text}'
        except requests.exceptions.RequestException as e:
            error = str(e)

    return render_template_string(
        PAGE_TEMPLATE,
        response=response,
        error=error,
        company=company,
        tweet_type=tweet_type,
        message=message
    )

if __name__ == '__main__':
    app.run(debug=True, port=8080)
# This code is a simple Flask web application that serves as a frontend for a tweet generator.