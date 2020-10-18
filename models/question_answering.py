from transformers import AutoModelForQuestionAnswering, pipeline
import streamlit as st


#@st.cache(allow_output_mutation=True, show_spinner=True)
def load_model(model_name):
    return pipeline("question-answering", 
                    model=model_name, 
                    tokenizer=model_name)


def question_answering():
    
    st.title("Question answering")
    input_text = st.text_area("Enter your text in the box below", height=250)
    question = st.text_area("Enter your question in the box below", height=100)
    model_name  = st.selectbox("Choose model",
        ["henryk/bert-base-multilingual-cased-finetuned-dutch-squad2",
         "distilbert-base-cased-distilled-squad"])
    
    pipeline_model = load_model(model_name=model_name)
    
    if  st.button("Run question answering"):
        
        answer = pipeline_model({"context": input_text,
                                "question": question})

        st.write("Answer:")
        st.success(answer["asnwer"])
        st.write("Confidence level:")
        st.success("{:.2f}".format(answer["score"]))


