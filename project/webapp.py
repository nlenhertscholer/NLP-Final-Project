# webapp.py
# by Nesta Lenhert-Scholer
#
# Backend for the simple website.
# Takes in data from the form and generates text depending on the options

from flask import Flask, render_template, request, url_for
import rnn
import torch

app = Flask(__name__)

# Paths for pre-trained models
MODEL = "models/"
WINE = MODEL + "wine_model_final.pth"
JOKE = MODEL + "jokes_model_final.pth"
JOKES = [MODEL + "jokes_prev_model/jokes_model_e1.pth",
         MODEL + "jokes_prev_model/jokes_model_e7.pth",
         MODEL + "jokes_prev_model/jokes_model_e13.pth",
         MODEL + "jokes_prev_model/jokes_model_e39.pth",
         JOKE]
WINES = [MODEL + "wine_model_e0.pth", MODEL + "wine_model_e2.pth",
         MODEL + "wine_model_e6.pth", MODEL + "wine_model_e8.pth",
         WINE]

# Wine/Joke data
WINE_DATA = "wine_data/wine_reviews_short.txt"
JOKE_DATA = "jokes/jokes.txt"


@app.route('/', methods=["GET", "POST"])
def hello_world():
    """Main index page of the site
    :return html_template taking in the generated text, if any"""

    text = ""
    if request.method == "POST":

        # Retrieve form data
        model = request.form.get("model_type")
        history = False
        if request.form.get("history"):
            history = True
        seed_text = request.form.get("seed-text")
        if seed_text == "":
            seed_text = "<"

        # Load the data
        datapath = WINE_DATA if model == "wine" else JOKE_DATA
        seq, itoc, ctoi = rnn.load_data(datapath)

        if history:
            # Generate text from the same model at different points in training
            filepaths = WINES if model == "wine" else JOKES
            models = []
            for filepath in filepaths:
                models.append(torch.load(filepath))
            for model in models:
                text += model.generate_text(ctoi, itoc,
                                            seq.shape[1], start_phrase=seed_text)
                text += "\n\n"
        else:
            # Generate text from the best model of the given type
            path = WINE if model == "wine" else JOKE
            model = torch.load(path)
            text += model.generate_text(ctoi, itoc,
                                        seq.shape[1], start_phrase=seed_text)

    return render_template("main.html", text=text)
