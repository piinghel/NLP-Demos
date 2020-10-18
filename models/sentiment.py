import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model_dir = "distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline("sentiment-analysis", tokenizer=tokenizer, model=model, return_all_scores=return_all_scores)

def predict(model, input_text, model_name="model name"):
    sentiment = model(input_text)
    df = pd.DataFrame(sentiment[0])
    df.rename(columns={"label": "Label", "score":"Confidence"}, inplace=True)
    df["Model"] = model_name
    return df[["Model", "Label", "Confidence"]]



def sentiment_analyzer():
    st.title("Sentiment analysis")
    # input
    example_text = """We are asking our stockholders to approve an amendment and restatement of our 2012 ESPP to increase the share reserve by 2,000,000 shares. The Board recommends a vote FOR this proposal because our employee stock purchase program is an important employee benefit and is essential to attracting, retaining and motivating our employees."""
    input_text = st.text_area("Enter your text here below.", example_text, height=125)

    # get model directories
    sample_dirs = """FinBERT:ipuneetrathore/bert-base-cased-finetuned-finBERT,
DistilBERT:distilbert-base-uncased-finetuned-sst-2-english, 
BERT-base-multilingual:nlptown/bert-base-multilingual-uncased-sentiment"""
    model_dir = st.text_area("""Enter the model directories here. Put model <MODEL NAME: DIRECTORY> (need to be from the sentiment class), different model directories are split by ' , '.""", sample_dirs, height=80).split(",")
    st.markdown("**Click on the following [link](https://huggingface.co/models?sort=modified&search=sentiment) to check out more models.**")

    if st.button("Run sentiment analysis"):
        with st.spinner('Loading models...'):
            # load models
            models = {m.split(":")[0]:load_model(model_dir=m.split(":")[1]) for m in model_dir}
        with st.spinner('Classifying...'):
            # predict
            out = [predict(model=v, input_text=input_text, model_name=k) for k, v in models.items()]        
            df = pd.concat(out).set_index("Label")
            st.write(df)