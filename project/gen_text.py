# gen_text.py
# by Nesta Lenhert-Scholer
#
# File loads the given data and trains the character-level RNN on the data

import rnn
import torch

from sklearn.model_selection import train_test_split

NUM_LAYERS = 1
EMB_SIZE = 128
HID_SIZE = 256
PAD_CHAR = '#'
JOKES = "jokes/jokes.txt"
WINE = "wine_data/wine_reviews_short.txt"

dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'

# Change this to JOKES to train on and generate jokes
filename = JOKES

seq, itoc, ctoi = rnn.load_data(filename)

if filename == JOKES:
    X_train, X_test = train_test_split(seq, test_size=0.25, shuffle=False)
else:
    # Decrease the text set even more to decrease training time
    X_train, X_test = train_test_split(seq, test_size=0.8, shuffle=False)
    X_test, _ = train_test_split(X_test, test_size=0.98, shuffle=False)

charnn = rnn.CharNN(len(itoc), num_layers=NUM_LAYERS, emb_size=EMB_SIZE, hid_size=HID_SIZE,
                    pad_idx=ctoi[PAD_CHAR], max_len=seq.shape[1], device=dev)

charnn.train_model(X_train, X_test, ctoi, itoc)

# Uncomment to save the model
# if filename == JOKES:
#     charnn.save_model("drive/My Drive/NLP_Final_Project_Data/models/jokes_model_final.pth")
# else:
#     charnn.save_model("drive/My Drive/NLP_Final_Project_Data/models/wine_model_final.pth")

# Uncomment the following line and comment out the previous lines (up to the declaration of charnn) to use a pre-trained
# Charnn model
# charnn = torch.load("drive/My Drive/NLP_Final_Project_Data/models/wine_model_final.pth")

print("------------Done Training-------------")

# Generate sentences based on a given seed
if filename == JOKES:
    start_seq = ["*Knock knock*", "What did", "Why did", "I heard"]
else:
    start_seq = ["This red wine", "This white wine", "Lovely", "Terrible"]

for start_s in start_seq:
    print(charnn.generate_text(ctoi, itoc, seq.shape[1], start_phrase=start_s))
    print()
