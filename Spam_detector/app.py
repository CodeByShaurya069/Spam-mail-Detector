from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

# Sample dataset
data = [
    ("ham", "Hey, are we still meeting today?"),
    ("ham", "Please call me when you are free."),
    ("ham", "The homework is attached."),
    ("ham", "Can you send the notes from class?"),
    ("ham", "See you at 5 pm."),
    ("ham", "Thanks for your help yesterday."),
    ("ham", "I will reach the office in 10 minutes."),
    ("ham", "Let us discuss the project tomorrow."),
    ("ham", "Happy birthday! Have a great day."),
    ("ham", "Your meeting has been rescheduled."),

    ("spam", "Congratulations! You won a free prize. Click now."),
    ("spam", "Claim your cash reward today by clicking the link."),
    ("spam", "URGENT: Your account is suspended. Verify now."),
    ("spam", "Get rich fast with this amazing offer."),
    ("spam", "Lowest price insurance. Buy now!"),
    ("spam", "You have been selected for a gift card."),
    ("spam", "Limited offer. Act now to get free access."),
    ("spam", "Winner! Reply with your bank details to collect."),
    ("spam", "This is not a scam. Send money to receive more."),
    ("spam", "Free money waiting for you. Click this link now.")
]

df = pd.DataFrame(data, columns=["label", "text"])

# Machine Learning Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("clf", MultinomialNB())
])

model.fit(df["text"], df["label"])

# Store history
history = []

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            pred = model.predict([message])[0]
            proba = model.predict_proba([message])[0]

            confidence = round(max(proba) * 100, 2)

            result = "SPAM" if pred == "spam" else "NOT SPAM"

            history.insert(0, {
                "message": message,
                "result": result,
                "confidence": confidence
            })

            # Keep only last 5 predictions
            if len(history) > 5:
                history.pop()

        else:
            result = "Please enter an email message."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message,
        history=history
    )

if __name__ == "__main__":
    app.run(debug=True)