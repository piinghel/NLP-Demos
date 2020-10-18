# import libraries
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd


# intialize model
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model_dir="facebook/bart-large-mnli"):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline("zero-shot-classification", tokenizer=tokenizer, model=model)

# @st.cache(allow_output_mutation=True, show_spinner=False)
def predict(model,
              doc,
              labels,
              model_name,
              multilabel=True,
              max_length=1024,
              batch_size=8,
              include_labels=True):

    output = model(doc, labels, "This text is about {}.", multilabel)
    df = pd.DataFrame({model_name:output["scores"]}).T
    df.columns = output['labels']
    return df


def topics_identifier():

    st.title('Topics identifier')
    st.text("Identifies user provided topcis in the text. You can choose your own text and topics/theme.")

    # get text and topics
    example_text = """We are asking our stockholders to approve an amendment and restatement of our 2012 ESPP to increase the share reserve by 2,000,000 shares. The Board recommends a vote FOR this proposal because our employee stock purchase program is an important employee benefit and is essential to attracting, retaining and motivating our employees."""
    doc = st.text_area('Text', example_text,
                       key='sequence', height=125)
    topics = st.text_input(
        'Possible topics (separated by `,`)', "Environmental, Social, Governance, ESG", max_chars=1000)
    topics = list(
        set([x.strip() for x in topics.strip().split(',') if len(x.strip()) > 0]))

    # allow multiple topics to be correct (default is true)
    multi_topics = st.checkbox('Allow multiple correct topics', value=True)

    # get model directories
    sample_dirs = """Bart MNLI:facebook/bart-large-mnli,
Bart MNLI + Yahoo Answers:joeddav/bart-large-mnli-yahoo-answers, 
XLM Roberta XNLI (cross-lingual):joeddav/xlm-roberta-large-xnli"""
    model_dir = st.text_area("""Enter the model directories here. Put model <MODEL NAME: DIRECTORY>, different model directories are split by ' , '.""", sample_dirs, height=80).split(",")
    st.markdown("**Click on the following [link](https://huggingface.co/models) to check out more models.**")
    
    # model configurations
    #st.markdown('**Model Configurations**')
    # max_length = st.slider("Select max lenght of the text", min_value=1, max_value=512*4, value=1024, step=1)
    # batch_size = st.slider("Select batch size", min_value=8, max_value=512, value=8, step=8)
    
    
    if st.button("Run topic identifier"):
            # load models
            with st.spinner('Loading models...'):
                models = {m.split(":")[0]:load_model(model_dir=m.split(":")[1]) for m in model_dir}
            # make predictions
            with st.spinner('Classifying...'):
                preds = [predict(model=v, doc=doc, labels=topics, multilabel=multi_topics, model_name=k) for k, v in models.items()]        
                df = pd.concat(preds)
                st.write(df)    

       






