from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import string
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

def get_text(msg):
    input_text = msg
    input_text = [input_text]
    df_input = pd.DataFrame(input_text, columns=['questions'])
    return df_input

model = load_model(r'C:\Users\Karan\Desktop\CHATBOT_OG\BVCOE chatbot\model-v1.keras')
tokenizer_t = joblib.load(r'C:\Users\Karan\Desktop\CHATBOT_OG\BVCOE chatbot\tokenizer_t.pkl')
vocab = joblib.load(r'C:\Users\Karan\Desktop\CHATBOT_OG\BVCOE chatbot\vocab.pkl')

def tokenizer(entry): 
    tokens = entry.split() 
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    tokens = [re_punc.sub('', w) for w in tokens] 
    tokens = [word for word in tokens if word.isalpha()] 
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens] 
    tokens = [word.lower() for word in tokens if len(word) > 1] 
    return tokens

def remove_stop_words_for_input(tokenizer, df, feature): 
    doc_without_stopwords = [] 
    entry = df[feature][0] 
    tokens = tokenizer(entry) 
    doc_without_stopwords.append(' '.join(tokens)) 
    df[feature] = doc_without_stopwords
    return df

def encode_input_text(tokenizer_t, df, feature): 
    t = tokenizer_t
    entry = [df[feature][0]] 
    encoded = t.texts_to_sequences(entry) 
    padded = pad_sequences(encoded, maxlen=16, padding='post') 
    return padded

def get_pred(model, encoded_input): 
    pred = np.argmax(model.predict(encoded_input), axis=-1) 
    return pred

def bot_precausion(df_input, pred): 
    words = df_input.questions[0].split() 
    if len([w for w in words if w in vocab]) == 0: 
        pred = 1
    return pred

def get_response(df2, pred):
    if isinstance(pred, np.ndarray):
        pred = pred.item()  # Convert ndarray to scalar if it's a single value
    
    # Ensure pred is a valid key for groupby
    if pred not in df2['labels'].values:
        raise ValueError(f"Prediction {pred} not found in labels")
    
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0, upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]

app = Flask(__name__) 

@app.route("/") 
def home(): 
    return render_template("head.html", template_folder='templates') 

@app.route("/get") 
def get_bot_response(): 
    userText = request.args.get('msg') 
    df_input = get_text(userText) 
    df2 = pd.read_csv(r"C:\Users\Karan\Desktop\CHATBOT_OG\BVCOE chatbot\response.csv") 
    df_input = remove_stop_words_for_input(tokenizer, df_input, 'questions') 
    encoded_input = encode_input_text(tokenizer_t, df_input, 'questions') 
    pred = get_pred(model, encoded_input) 
    pred = bot_precausion(df_input, pred) 
    response = get_response(df2, pred) 
    return response

if __name__ == "__main__": 
    app.run(debug=True)
