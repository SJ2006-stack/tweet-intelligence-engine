from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder


sheet_id = "1JcESl7qCCBvS6xpWMZplhCXunvmkcNU_"
gid = "95254209"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

df = pd.read_csv(url)


encoder = LabelEncoder()

df['company_encoded'] = encoder.fit_transform(df['inferred company'])
df['username_encoded'] = encoder.fit_transform(df['username'])
df.dropna(subset=['content', 'username', 'inferred company', 'likes'], inplace=True)
df.fillna({'media':'no_media'}, inplace=True)
df['has_media'] = df['media'].apply(lambda x: x != 'no_media')
df['content'] = df['content'].astype(str).str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['content', 'username', 'inferred company', 'likes'], inplace=True)
df.fillna({'media':'no_media'}, inplace=True)
df['has_media'] = df['media'].apply(lambda x: x != 'no_media')
df['content'] = df['content'].astype(str).str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.day_name()
df['day_of_week_encoded'] = encoder.fit_transform(df['day_of_week'])
df['word_count'] = df['content'].apply(lambda x: len(x.split()))
df['char_count'] = df['content'].apply(len)
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['polarity'] = df['content'].apply(get_polarity)
df['sentiment'] = df['polarity'].apply(get_sentiment)

def count_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

df['emoji_count'] = df['content'].apply(count_emojis)

def has_hashtag(text):
    # Returns 1 if at least one hashtag is present, else 0
    return 1 if re.search(r'#\w+', str(text)) else 0

df['has_hashtag'] = df['content'].apply(has_hashtag)

def has_url(text):
    url_regex = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return 1 if re.search(url_regex, str(text)) else 0

df['has_url'] = df['content'].apply(has_url)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['content'])

df['tfidf_mean'] = np.asarray(tfidf_matrix.mean(axis=1)).flatten()

company_avg_likes = df.groupby('inferred company')['likes'].mean().rename('company_avg_likes')
df = df.merge(company_avg_likes, on='inferred company', how='left')




# df.head() '''already checked'''

X = df[['word_count', 'char_count', 'has_media', 'hour','sentiment','company_encoded','emoji_count','has_url','has_hashtag','tfidf_mean','company_avg_likes']]
y = df['likes']

sentiment_encoded = pd.get_dummies(df['sentiment'], prefix='sentiment', drop_first=True)
X = X.drop('sentiment', axis=1)
X = pd.concat([X, sentiment_encoded], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestRegressor()
model1.fit(X_train, y_train)

rmse=[]
preds1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, preds1)
rmse_RFR = np.sqrt(mse1)
rmse.append(rmse_RFR)

from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 2000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2)
}







model2 = LinearRegression()
model2.fit(X_train, y_train)
preds2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, preds2)
rmse_LR = np.sqrt(mse2)
rmse.append(rmse_LR)

'''model3 = XGBRegressor(n_estimators=121, max_depth=4, learning_rate=0.13349630192554332)
model3.fit(X_train, y_train)
preds3 = model3.predict(X_test)
mse3 = mean_squared_error(y_test, preds3)
rmse_XGB = np.sqrt(mse3)
rmse.append(rmse_XGB)'''
'''random_search = RandomizedSearchCV(estimator=model3, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)'''

base_models1 = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]


model4 = StackingRegressor(estimators=base_models1, final_estimator=model2)
model4.fit(X_train, y_train)
preds4 = model4.predict(X_test)
mse4 = mean_squared_error(y_test, preds4)
rmse_SR = np.sqrt(mse4)
rmse.append(rmse_SR)

#n_estimators=148, max_depth=3, learning_rate=0.011413261043943482


# 4403.512089210708(random forest)
# 3207.130025427719(XGBOOST)(Best parameters: {'learning_rate': np.float64(0.13349630192554332), 'max_depth': 4, 'n_estimators': 121})
'''scores = cross_val_score(
    XGBRegressor(),
    X, y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)
rmse_XGB = -scores
print("RMSE per fold:", rmse_XGB)
print("Mean RMSE:", rmse_XGB.mean())'''
# no point in doing cross_val_score as we already have the best parameters for XGBoost (RMSE per fold: [3702.74243164 5376.37402344 4365.22509766 3278.94238281 4252.16308594]
#Mean RMSE: 4195.089404296875
#[3702.74243164 5376.37402344 4365.22509766 3278.94238281 4252.16308594])

# Okay, now I tried ensembled and fuck, rn the lowest rmse right is 3064.7213105488117 (with final_estimator = linear_regressor)
#(with final_estimator(XGB)=3242.4803777355382)
#(with final_estimator(random_forest)=4080.2480659755847

#print(rmse_SR)

joblib.dump(model4, 'like_predictor.pkl', compress=3)
