# NLP-Final-Project
Final project for NLP at the University of Chicago during the Summer of 2020

The [project_emb](./project_emb) directory contains previous work done before the text generation project. There is a readme included in that folder for a description of that project.

Inside of [project](./project) is the source code and data for text generation using an RNN. The description of the source code files is as follows:
- [webapp.py](./project/webapp.py) - 69 lines of code. Used as the backend for a simple website to drive the text generation
- [runwebapp.sh](./project/runwebapp.sh) - 6 lines of code. Used to start the webapp locally
- [rnn.py](./project/rnn.py) - 303 lines of code. File containing the RNN class and a function to help load the data
- [gen_text.py](./project/gen_text.py) - 59 lines of code. Driver to train RNN model and generate a few example sentences
- [Pipfile](./project/Pipfile) - Description of python virtual environment. Need to use *pipenv* to generate the environment
- [wine_data/data_clean.py](./project/wine_data/data_clean.py) - 24 lines of code. Convert wine csv files into a single text file
- [templates/main.html](./project/templates/main.html) - 43 lines of code. HTML describing web page
- [jokes/clean_data.py](./project/jokes/clean_data.py) - 23 lines of code. Convert the jokes csv file into a single text file

The rest of the files are data and generated text. [wine_data](./project/wine_data) and [jokes](./project/jokes) hold data and text generation for both the wine and jokes data respectively.

## Notes
The website is able to be rendered, however, the loading of models might not work correctly as inteded. This is to be looked at in the future and hopefully fixed. 

## References
All code in this project was written by me. I would like to acknowledge sources that helped me throughout this project:
- [How to Clean Text for Machine Learning with Python](https://machinelearningmastery.com/clean-text-machine-learning-python/)
- [NLP using RNN — Can you be the next Shakespeare?](https://medium.com/analytics-vidhya/nlp-using-rnn-can-you-be-the-next-shakespeare-27abf9af523)
- [NLP FROM SCRATCH: GENERATING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [Character Level RNN](https://github.com/pskrunner14/char-level-rnn)
- [Character-Level Language Model](https://towardsdatascience.com/character-level-language-model-1439f5dd87fe)
