from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for predicting fake news
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"]  # Get input from form
        input_data = vectorizer.transform([news_text])  # Convert text to numerical form
        prediction = model.predict(input_data)[0]  # Get prediction (0 = Fake, 1 = Real)
        
        result = "Real News" if prediction == 1 else "Fake News"
        return render_template("result.html", news_text=news_text, result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
