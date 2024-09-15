import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Define your private model's Hugging Face path and token
model_name_or_path = "LLLLuke/autotrain-0a8wb-l6joy"
token = "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"

# Load the model and tokenizer with the authentication token
tokenizer = MarianTokenizer.from_pretrained(model_name_or_path, use_auth_token=token)
model = MarianMTModel.from_pretrained(model_name_or_path, use_auth_token=token)

# Function to perform translation
def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output_tokens = model.generate(**inputs)
    translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translation

# Streamlit UI
st.title("Machine Translation App")

# Text input for the user to enter a sentence
input_text = st.text_input("Enter the text you want to translate:")

# Add a button to trigger the translation
if st.button("Translate"):
    if input_text:
        # Translate the input text
        translated_text = translate(input_text, model, tokenizer)
        # Display the translation
        st.write("Translated Text:")
        st.success(translated_text)
    else:
        st.warning("Please enter some text to translate.")
