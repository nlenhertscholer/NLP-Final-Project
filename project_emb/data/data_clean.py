import pandas as pd
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

# Read in the data
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add in the classifications
fake_df["class"] = 0
true_df["class"] = 1

# Remove publication information since that is only available in the true dataset
text = []
for t in true_df.text.values:
    split_t = t.split("-", maxsplit=1)
    text.append("-".join(split_t[1:]) if "(Reuters)" in split_t[0] else "-".join(split_t))
true_df["text"] = text

# Combine into one array
df = pd.concat([fake_df, true_df])

# Remove unnecessary columns and combine title into the text column
WANTED_COLS = ["text", "class"]
df["text"] = df["title"] + " " + df["text"]
df.drop([col for col in list(df.columns) if col not in WANTED_COLS], axis=1, inplace=True)

# Clean the text to remove unnecessary words and tokenize it
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
y = df["class"].values
X = []
for t in df["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(t)
    for s in sentences:
        s = s.lower()
        tokens = tokenizer.tokenize(s)
        filtered_words = [word.strip() for word in tokens if word not in stop_words and len(word) > 1]
        tmp.extend(filtered_words)
    X.append(" ".join(tmp))

df = pd.DataFrame(list(zip(X, y)), columns=["text", "class"])

# Split into train, validate, and test files
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

train.to_csv("train.csv", index=False)
validate.to_csv("validate.csv", index=False)
test.to_csv("test.csv", index=False)
