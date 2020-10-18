import streamlit as st
from ktrain import text

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    return text.TransformerSummarizer()

def text_summarizer():
    st.title("Text summarization")
    example_text = """World War II (WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. In a state of total war, directly involving more than 100 million people from more than 30 countries, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities. Tens of millions of people died due to genocides (including the Holocaust), premeditated death from starvation, massacres, and disease. Aircraft played a major role in the conflict, including in the use of strategic bombing of population centres, and the only uses of nuclear weapons in war.
World War II is generally considered to have begun on 1 September 1939, with the invasion of Poland by Germany and subsequent declarations of war on Germany by France and the United Kingdom. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union partitioned and annexed territories of their European neighbours: Poland, Finland, Romania and the Baltic states. Following the onset of campaigns in North Africa and East Africa, and the Fall of France in mid-1940, the war continued primarily between the European Axis powers and the British Empire, with war in the Balkans, the aerial Battle of Britain, the Blitz, and the Battle of the Atlantic. On 22 June 1941, Germany led the European Axis powers in an invasion of the Soviet Union, opening the largest land theatre of war in history and trapping the Axis, crucially the German Wehrmacht, in a war of attrition.
Japan, which aimed to dominate Asia and the Pacific, was at war with the Republic of China by 1937. In December 1941, Japan launched a surprise attack on the United States as well as European colonies in East Asia and the Pacific. Following an immediate US declaration of war against Japan, supported by one from the UK, the European Axis powers declared war on the United States in solidarity with their ally. Japan soon captured much of the Western Pacific, but its advances were halted in 1942 after losing the critical Battle of Midway; later, Germany and Italy were defeated in North Africa and at Stalingrad in the Soviet Union. Key setbacks in 1943—including a series of German defeats on the Eastern Front, the Allied invasions of Sicily and Italy, and Allied offensives in the Pacific—cost the Axis its initiative and forced it into strategic retreat on all fronts. In 1944, the Western Allies invaded German-occupied France, while the Soviet Union regained its territorial losses and turned towards Germany and its allies. During 1944 and 1945, Japan suffered reversals in mainland Asia, while the Allies crippled the Japanese Navy and captured key Western Pacific islands."""
    input_text = st.text_area("Enter your text below", example_text, height=250)
    
    
    if st.button("Run"):
        with st.spinner('Loading models...'):
            model = load_model()
        with st.spinner('Summarizing...'):
            summary = model.summarize(input_text)
            st.success(summary)



        
