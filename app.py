import streamlit as st
from collections import defaultdict
import tqdm
import transformers
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random, glob


@st.cache_data
def load_data():
    return pd.read_csv("MassiveDatasetValidationData.csv")


def reload_example_text_data(language):
    random_id = random.choice(val_data["id"])
    tempdf = val_data[val_data["id"] == random_id]
    tempdf = tempdf[["iso", "text", *selected_tokenizers]]
    tempdf = tempdf[tempdf["iso"] == language]
    tempdf.set_index("iso", inplace=True)
    tempdf.columns = ["Text"] + [f"Num Tokens ({t})" for t in selected_tokenizers]
    st.session_state.examplesdf = tempdf


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
    st.header("Comparing Tokenizers")
    link = "This project compares the tokenization length for different tokenizers. Some tokenizers may result in significantly more tokens than others for the same text."
    st.markdown(link)

    st.header("Data Visualization")
    st.subheader("Tokenizers")
    selected_tokenizers = st.multiselect(
        "Select tokenizers",
        options=tokenizer_names_to_test,
        default=["openai/gpt4", "Xenova/gpt-4o", "Xenova/claude-tokenizer"],
        max_selections=6,
        label_visibility="collapsed",
    )

    st.subheader("Data")
    with st.spinner("Loading dataset..."):
        val_data = load_data()
    st.success(f"Data loaded: {len(val_data)}")

    with st.expander("Data Source"):
        st.write(
            "The data in this figure is the validation set of the [Amazon Massive](https://huggingface.co/datasets/AmazonScience/massive/viewer/af-ZA/validation) dataset, which consists of 2033 short sentences and phrases translated into 51 different languages. Learn more about the dataset from [Amazon's blog post](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)"
        )

    st.subheader("Language")
    language_options = sorted(val_data.lang.unique())
    default_language_index = (
        language_options.index("English") if "English" in language_options else 0
    )
    selected_language = st.selectbox(
        "Select language",
        options=language_options,
        index=default_language_index,
        label_visibility="collapsed",
    )

    st.subheader("Figure")
    selected_figure = st.radio(
        "Select figure type",
        options=["Boxplot", "Histogram", "Scatterplot"],
        index=0,
        label_visibility="collapsed",
    )

    st.header("Example Text")
    with st.spinner("Loading example text..."):
        reload_example_text_data(selected_language)
    st.table(st.session_state.examplesdf)
    st.button("Reload", on_click=reload_example_text_data, args=(selected_language,))

    tokenizer_to_num_tokens = defaultdict(list)
    for _, row in tqdm.tqdm(val_data.iterrows(), total=len(val_data)):
        text = row["text"]
        for tokenizer_name in selected_tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            num_tokens = len(tokenizer(text)["input_ids"])
            tokenizer_to_num_tokens[tokenizer_name].append(num_tokens)

    if selected_figure == "Boxplot":
        fig = go.Figure()
        for tokenizer_name in selected_tokenizers:
            fig.add_trace(
                go.Box(y=tokenizer_to_num_tokens[tokenizer_name], name=tokenizer_name)
            )
        fig.update_layout(
            title=f"Distribution of Number of Tokens for Selected Tokenizers",
            xaxis_title="Tokenizer",
            yaxis_title="Number of Tokens",
        )
        st.plotly_chart(fig)
    elif selected_figure == "Histogram":
        fig = make_subplots(
            rows=len(selected_tokenizers), cols=1, subplot_titles=selected_tokenizers
        )
        for i, tokenizer_name in enumerate(selected_tokenizers):
            fig.add_trace(
                go.Histogram(
                    x=tokenizer_to_num_tokens[tokenizer_name], name=tokenizer_name
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(
            height=200 * len(selected_tokenizers),
            title_text="Histogram of Number of Tokens",
        )
        st.plotly_chart(fig)
    elif selected_figure == "Scatterplot":
        df = pd.DataFrame(tokenizer_to_num_tokens)
        fig = px.scatter_matrix(
            df,
            dimensions=selected_tokenizers,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_layout(
            title=f"Scatterplot Matrix of Number of Tokens for Selected Tokenizers",
            width=800,
            height=800,
        )
        st.plotly_chart(fig)
