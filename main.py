#from pydantic import BaseModel
#from fastapi.staticfiles import StaticFiles
#from pydantic.main import BaseModel
import uvicorn
from fastapi import Request, FastAPI
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_hub as hub
import bert
import nltk
import os
import re
from nltk.tokenize import word_tokenize
import nltk
#from fastapi.responses import HTMLResponse
#from fastapi.templating import Jinja2Templates
#nltk.download('punkt')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading model
model = tf.keras.models.load_model(
    'model_new_version/content/model')


#class get_sentiment(BaseModel):
#    text: str


# Loading tokenizer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


def data_cleaning(text):
    # Case folding
    text = text.lower()

# Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

# Remove stop words from several languages
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words(
        ['english', 'portuguese', 'french', 'spanish', 'russian', 'italian', 'german'])]
    text = (" ").join(tokens_without_sw)

# Tokenize text with bert
    tokenizing_bert = tokenizer.tokenize(text)

# Convert tokens to id with bert
    text = tokenizer.convert_tokens_to_ids(tokenizing_bert)

    return text


app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the game's reviews"
)

#app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")


@app.get("/")
# , response_class=HTMLResponse)
async def home():
    # (request: Request):
    # return templates.TemplateResponse("index.html", {"request": request})
    return("Hello World")


@app.post("/predict")
def predict(text: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    """
# Apply the cleaning data
    cleaned_text = data_cleaning(text)

# Perform prediction
    prediction = model.predict([cleaned_text])
    prediction = int(prediction)

# Show result
    if prediction > 0.5:
        prediction = "Positif (Produk direkomendasikan)"
    else:
        prediction = "Negatif (Produk tidak direkomendasikan)"

    return {"review": text, "prediction": prediction}

# return templates.TemplateResponse("result.html", {"request": request})
