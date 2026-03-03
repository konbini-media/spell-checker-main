from openai import OpenAI
import streamlit as st

st.set_page_config(page_title='Konbini Correction Tool', page_icon=None, layout="wide")


st.image('https://upload.wikimedia.org/wikipedia/fr/thumb/0/0a/Logo-konbini.svg/langfr-1280px-Logo-konbini.svg.png', width=78)

st.write(
    """
    # Konbini Correction Tool

    Merci d'utiliser la bonne page en fonction du use case :    
    """
)

st.page_link("pages/Konbini_Articles.py", label="Corriger un article")
st.page_link("pages/Konbini_Captions.py", label="Corriger une caption")


st.info(
    """
    Pour toutes questions : contacter Johan Shim (johan.shim@konbini.com)
    """
)