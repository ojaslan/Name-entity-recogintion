import streamlit as st
from transformers import pipeline


# Load NER model
@st.cache_resource
def load_ner_model():
    return pipeline("ner", grouped_entities=True)


ner_model = load_ner_model()

# Set the title with custom style
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Named Entity Recognition </h1>",
    unsafe_allow_html=True
)

# Set up text input with a description
st.write(
    "<p style='text-align: center;'>Enter your text below for entity recognition.</p>",
    unsafe_allow_html=True
)

# Center the input text area
text_input = st.text_area(
    "Text Input",
    placeholder="Type your text here...",
    height=200
)

# Customize the button and center it
button_style = """
    <style>
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: large;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Button to trigger NER model
if st.button("Recognize Entities"):
    if text_input:
        with st.spinner("Processing..."):
            entities = ner_model(text_input)

        if entities:
            st.subheader("Named Entities")

            # Loop through entities and display with improved visibility
            for entity in entities:
                entity_html = f"""
                    <div style="background-color: #333333; border-radius: 8px; padding: 10px; margin: 5px 0;">
                        <strong style="color: #ff9800;">Entity:</strong> {entity['word']} 
                        <br>
                        <strong style="color: #03a9f4;">Type:</strong> {entity['entity_group']} 
                        <br>
                        <strong style="color: #8bc34a;">Confidence:</strong> {entity['score']:.2f}
                    </div>
                """
                st.markdown(entity_html, unsafe_allow_html=True)
        else:
            st.write("No named entities found in the text.")
    else:
        st.error("Please enter some text for entity recognition.")
