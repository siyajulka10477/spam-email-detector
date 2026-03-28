import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("spam.csv")

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = data['message']
y = data['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test with custom input
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)
    return "🚨 Spam" if result[0] == 1 else "✅ Not Spam"

msg = input("Enter your message: ")

# Predict
print(predict_message(msg))