from models.topics_identifier import topics_identifier
from models.sentiment import sentiment_analyzer
from models.text_summarizer import text_summarizer
from models.question_answering import question_answering
import streamlit as st

def main():
    st.sidebar.title("NLP applications")
    mode  = st.sidebar.selectbox("Choose your demo",
        ["Topic classification","Sentiment analysis"])

    if mode == "Topic classification":
        topics_identifier()

    elif mode == "Sentiment analysis":
        sentiment_analyzer()

    # elif mode == "Text summarization":
    #    text_summarizer()
    
    # elif mode == "Question answering":
    #    question_answering()

if __name__ == "__main__":
    main()