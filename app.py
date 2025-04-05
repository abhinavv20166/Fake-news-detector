from flask import Flask, render_template, request
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Set of stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess input text by tokenizing, converting to lowercase, and removing stopwords."""
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for predicting fake news
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"]
        
        # Check for empty input
        if not news_text:
            return render_template("result.html", news_text=news_text, result="Error: Please enter some text!")
        
        # Preprocess the input text
        news_text = preprocess_text(news_text)
        
        # Convert text to numerical form
        input_data = vectorizer.transform([news_text])
        
        # Get prediction (0 = Fake, 1 = Real)
        prediction = model.predict(input_data)[0]
        
        result = "Real News" if prediction == 1 else "Fake News"
        return render_template("result.html", news_text=news_text, result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False for production
