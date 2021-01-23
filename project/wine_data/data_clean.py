# data_clean.py
# by Nesta Lenhert-Scholer
#
# Prepares the wine data from the wine csv file.

import unicodedata      # Used to clean the data from unicode characters
import pandas as pd

START_CHAR = '<'
END_CHAR = '>'

df1 = pd.read_csv("wine_data/winemag-data_first150k.csv")
df2 = pd.read_csv("wine_data/winemag-data-130k-v2.csv")
df = pd.concat([df1["description"], df2["description"]], ignore_index=True)

for i, review in enumerate(df.values):
    r = START_CHAR + unicodedata.normalize("NFKD", review) + END_CHAR
    df[i] = r

df.drop_duplicates(inplace=True)

with open("wine_data/wine_reviews.txt", "w", encoding="utf-8") as f:
    for review in df.values:
        f.write(review)
