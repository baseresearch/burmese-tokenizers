import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")


def reload_example_text_data(selected_language, selected_tokenizers):
    tempdf = val_data[val_data["lang"] == selected_language]
    random_sample = tempdf.sample(n=1)
    selected_text = random_sample["text"].iloc[0]
    random_sample = random_sample[selected_tokenizers]
    random_sample.columns = [f"{tokenizer}" for tokenizer in selected_tokenizers]
    st.session_state.examplesdf = random_sample
    return selected_text


val_data = load_data()

tokenizer_names_to_test = [
    "openai/gpt4",
    "Xenova/gpt-4o",
    "Xenova/claude-tokenizer",
    "CohereForAI/aya-101",
    "meta-llama/Meta-Llama-3-70B",
    "mistralai/Mixtral-8x22B-v0.1",
    "google/gemma-7b",
    "facebook/nllb-200-distilled-600M",
    "xlm-roberta-base",
    "bert-base-uncased",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bigscience/bloom",
    "StabilityAI/stablelm-base-alpha-7b",
    "google/flan-t5-base",
    "facebook/mbart-large-50",
    "EleutherAI/gpt-neox-20b",
]

with st.sidebar:

    st.title("Tokenizer Comparisons")
    st.write(
        """
    Explore the performance of various tokenizers on the Burmese language. This tool visualizes how different tokenizers process the same Burmese text, highlighting disparities in tokenization.
    
    This project is inspired by the insights from "All languages are NOT created (tokenized) equal!" Read more about it in the original article on [Art Fish Intelligence](https://www.artfish.ai/p/all-languages-are-not-created-tokenized).
    """
    )

    all_tokenizers = st.checkbox("Select All Tokenizers")
    if all_tokenizers:
        selected_tokenizers = tokenizer_names_to_test
    else:
        selected_tokenizers = st.multiselect(
            "Select tokenizers",
            options=tokenizer_names_to_test,
            default=[
                "openai/gpt4",
                "Xenova/gpt-4o",
                "CohereForAI/aya-101",
                "Xenova/claude-tokenizer",
            ],
            label_visibility="collapsed",
        )
    links = [
        (
            f"[{tokenizer_name}](https://huggingface.co/{tokenizer_name})"
            if tokenizer_name != "openai/gpt4"
            else f"[{tokenizer_name}](https://github.com/openai/tiktoken)"
        )
        for tokenizer_name in selected_tokenizers
    ]
    link = "Tokenized using " + ", ".join(links)
    st.markdown(link, unsafe_allow_html=True)

selected_text = reload_example_text_data("Burmese", selected_tokenizers)
st.subheader(f"**Sampled Text:** `{selected_text}`")
st.subheader("Number of Tokens")
st.table(st.session_state.examplesdf)

# Create a distribution plot for token density across selected tokenizers
import plotly.figure_factory as ff

if selected_tokenizers:
    # Collecting data for all selected tokenizers
    hist_data = [val_data[tokenizer].dropna() for tokenizer in selected_tokenizers]

    # Creating the distplot with optional histogram
    fig = ff.create_distplot(
        hist_data, selected_tokenizers, show_hist=False, show_rug=False
    )
    fig.update_layout(
        title="Token Distribution Density",
        xaxis_title="Number of Tokens",
        yaxis_title="Density",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    tokenizer_to_num_tokens = {
        name: val_data[name].tolist() for name in selected_tokenizers
    }

    fig = go.Figure()
    for tokenizer_name in selected_tokenizers:
        fig.add_trace(
            go.Box(y=tokenizer_to_num_tokens[tokenizer_name], name=tokenizer_name)
        )
    fig.update_layout(title="Token Count Variability")
    st.plotly_chart(fig)
else:
    st.error(
        "No tokenizers selected. Please select at least one tokenizer to view the distribution plot."
    )
