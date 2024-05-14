import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
# Assume 'dataset.csv' has two columns: 'review' and 'sentiment'
data = pd.read_csv('IMDB Dataset.csv')

# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


data['cleaned_review'] = data['review'].apply(preprocess_text)
# Ensure the sentiment labels are numeric
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], data['sentiment'], test_size=0.2,
                                                    random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert labels to numpy array
y_train = np.array(y_train)
y_test = np.array(y_test)

# Model building using logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Optional: Deep learning approach
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],)), Dropout(0.5),
    Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_tfidf, y_train, epochs=5, batch_size=32, validation_data=(X_test_tfidf, y_test))

loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f'Accuracy: {accuracy}')

# Deployment using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    processed_review = preprocess_text(review)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model.predict(review_tfidf)
    return jsonify({'sentiment': 'positive' if prediction[0] >= 0.5 else 'negative'})


if __name__ == '__main__':
    app.run(debug=True)
