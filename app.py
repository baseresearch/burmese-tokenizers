import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random


@st.cache
def load_data():
    return pd.read_csv("dataset.csv")


def reload_example_text_data(language):
    random_id = random.choice(val_data["id"])
    tempdf = val_data[val_data["id"] == random_id]
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
    link = "This project compares the tokenization length for different tokenizers."
    st.markdown(link)
    selected_tokenizers = st.multiselect(
        "Select tokenizers",
        options=tokenizer_names_to_test,
        default=["openai/gpt4", "Xenova/gpt-4o", "Xenova/claude-tokenizer"],
        max_selections=6,
        label_visibility="collapsed",
    )

    for tokenizer_name in selected_tokenizers:
        if tokenizer_name == "openai/gpt4":
            link = "Tokenized using [tiktoken](https://github.com/openai/tiktoken)"
        else:
            url = f"https://huggingface.co/{tokenizer_name}"
            link = f"Tokenizer is available [on the HuggingFace hub]({url})"
        st.markdown(link, unsafe_allow_html=True)

    val_data = load_data()
    st.success(f"Data loaded: {len(val_data)}")

    language_options = sorted(val_data["lang"].unique())
    selected_language = st.selectbox(
        "Select language",
        options=language_options,
        index=language_options.index("English") if "English" in language_options else 0,
        label_visibility="collapsed",
    )

    selected_figure = st.radio(
        "Select figure type",
        options=["Boxplot", "Histogram", "Scatterplot"],
        index=0,
        label_visibility="collapsed",
    )

st.header("Example Text")
reload_example_text_data(selected_language)
st.table(st.session_state.examplesdf)
st.button("Reload", on_click=reload_example_text_data, args=(selected_language,))

tokenizer_to_num_tokens = {
    name: val_data[name].tolist() for name in selected_tokenizers
}

if selected_figure == "Boxplot":
    fig = go.Figure()
    for tokenizer_name in selected_tokenizers:
        fig.add_trace(
            go.Box(y=tokenizer_to_num_tokens[tokenizer_name], name=tokenizer_name)
        )
    fig.update_layout(title="Distribution of Number of Tokens for Selected Tokenizers")
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
    fig = px.scatter_matrix(df, dimensions=selected_tokenizers)
    fig.update_layout(
        title="Scatterplot Matrix of Number of Tokens for Selected Tokenizers"
    )
    st.plotly_chart(fig)
