# 🚀 Tweet Intelligence Engine



# 🐦 Tweet Intelligence Engine (T.I.E) 🚀🔥

Welcome to **Tweet Intelligence Engine (T.I.E)** — a cutting-edge ML-powered web app that predicts the *likes* a tweet might get based on its content, context, and metadata, and even generates compelling tweets tailored to your brand and topic.

---

## ⚡ Features

- **Data-driven like prediction:**  
  Uses a robust ensemble model (Stacking Regressor combining RandomForest, XGBoost, and Linear Regression) trained on rich tweet features:

  - Content features: word & char counts, TF-IDF score, sentiment polarity & category (positive/neutral/negative)
  - Metadata features: posting hour, presence of media, hashtags, URLs, emojis
  - Brand context: company encoding & average company likes
- **Smart Tweet Generator:**  
  Generate tweet text dynamically based on brand, tweet type (e.g., announcement, question, tip), and optional topic/message.
- **Interactive UI with modern, elegant theme:**  
  Built using Flask + Jinja2, featuring light/dark mode toggle with smooth animations and a beautiful Playfair Display font styling.
- **Robust preprocessing pipeline:**  
  Cleans input data, encodes categorical variables, extracts advanced features like emoji counts and sentiment analysis (via TextBlob).
- **Model persistence & compression:**  
  Serialized model (`like_predictor.pkl`) is compressed for efficient storage and fast loading.

---

## 🧰 Tech Stack & Libraries

- Python 3.8+
- Flask — for serving the web app
- scikit-learn — model building, LabelEncoding, feature engineering
- XGBoost — gradient boosted decision trees
- TextBlob — sentiment polarity and classification
- pandas, numpy — data processing
- joblib — model save/load with compression
- Jinja2 templating — dynamic HTML rendering

---

## 🚀 How It Works - The Pipeline

1. **Data ingestion:**  
   Loads tweet dataset from Google Sheets as CSV.

2. **Data cleansing & feature extraction:**  
   - Drops rows missing critical info
   - Encodes companies and users with `LabelEncoder`
   - Creates new features such as:
     - Sentiment polarity & sentiment category via TextBlob
     - Emoji counts via regex matching Unicode emoji ranges
     - Detects presence of hashtags & URLs
     - Computes TF-IDF mean scores on tweet content
     - Tags if media is attached
     - Extracts posting hour and day of week

3. **Model training:**  
   - Base learners: Random Forest Regressor, XGBoost Regressor
   - Final estimator: Linear Regression (for stacking ensemble)
   - Trained on engineered features to predict number of likes
   - Multiple models evaluated with RMSE metric; stacking ensemble yields best results

4. **Model deployment:**  
   Flask web app loads the compressed model, accepts user input from form,  
   generates a tweet (with SimpleTweetGenerator), extracts features from generated tweet,  
   and returns predicted likes along with tweet text.

---

## 🖥️ Web App Usage

- Open the app in your browser.
- Enter your **Company/Brand** name.
- Pick your tweet type (announcement, question, tip, etc.).
- Enter an optional **Message or Topic**.
- Specify if your tweet includes media.
- Choose the hour to post (0-23).
- Click **Predict & Generate**.
- See your generated tweet plus predicted likes score.

Interact with the smooth UI featuring playful emojis 🔥🚀 and toggle between light/dark modes for a stylish experience!

---

## 🛠️ Running Locally

1. Clone the repo (note: `like_predictor.pkl` is compressed for size)
2. Create a virtual environment and install dependencies:

```bash

git clone https://github.com/yourusername/tweet-intelligence-engine
cd tweet-intelligence-engine
pip install -r requirements.txt

```

## Run Flask app:

```bash

python app.py

```

4. Open `http://localhost:10000` in your browser.

---

## 📊 Model Performance Summary

| Model              | RMSE (Root Mean Squared Error) |
|--------------------|--------------------------------|
| Random Forest      | ~4403                          |
| Linear Regression  | ~3207                          |
| XGBoost (tuned)   | ~3207 (best params)            |
| Stacking (final)   | ~3064 (lowest RMSE)            |

---

## 🤝 Credits & Acknowledgements

- Built using [Scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io), and [TextBlob](https://textblob.readthedocs.io).
- Design inspired by modern UI patterns with smooth gradients and typography (Playfair Display).
- Thanks to the open-source community for model algorithms and NLP tools.
- Special Thanks to the CAIC and devclub guys for my first project.

---

## 🎉 Let’s make your tweets 🔥🚀 with **Tweet Intelligence Engine**

**Boost your social media impact by predicting likes and crafting catchy tweets tailored for your audience and brand!**

---

*Feel free to  contact me at me1240194@iitd.ac.in or shriyanshjaiswwal.xf@gmail.com*
*Hoping to work on more projects with CAIC and Devclub*
---

---
till we meet yet again....
---





