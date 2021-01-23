# clean_data.py
# by Nesta Lenhert-Scholer
#
# File prepares the csv joke data by adding start and stop characters to each joke.

import unicodedata      # Needed to remove some unicode data present in the text
import pandas as pd

START_CHAR = '<'
END_CHAR = '>'

df = pd.read_csv("shortjokes.csv", nrows=5000)

df = df["Joke"]
for i, joke in enumerate(df.values):
    r = START_CHAR + unicodedata.normalize("NFKD", joke) + END_CHAR
    df[i] = r

df.drop_duplicates(inplace=True)

with open("jokes.txt", "w", encoding="utf-8") as f:
    for joke in df.values:
        f.write(joke)
