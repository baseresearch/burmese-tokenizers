import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

with st.sidebar:
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
    selected_tokenizers = st.multiselect(
        "Select tokenizers",
        options=tokenizer_names_to_test,
        default=["openai/gpt4", "Xenova/gpt-4o"],
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

    language_options = sorted(val_data["lang"].unique())
    selected_language = st.selectbox(
        "Select language",
        options=language_options,
        index=language_options.index("English") if "English" in language_options else 0,
        label_visibility="collapsed",
    )

selected_text = reload_example_text_data(selected_language, selected_tokenizers)
st.subheader(f"**Sampled Text:** `{selected_text}`")
st.subheader("Number of Tokens")
st.table(st.session_state.examplesdf)

# Calculate metrics for each tokenizer
tokenizer_metrics = {}
for tokenizer in selected_tokenizers:
    tokens = val_data[tokenizer].dropna()
    median = np.median(tokens)
    min_tokens = np.min(tokens)
    max_tokens = np.max(tokens)
    std_dev = np.std(tokens)
    tokenizer_metrics[tokenizer] = {
        "Median": median,
        "Min": min_tokens,
        "Max": max_tokens,
        "Range": max_tokens - min_tokens,
        "Standard Deviation": std_dev,
    }

# Display metrics
st.subheader("Tokenizer Metrics")
st.json(tokenizer_metrics)

# Plot for top tokenizers by median token length
sorted_tokenizers = sorted(tokenizer_metrics.items(), key=lambda x: x[1]["Median"])
shortest_median = sorted_tokenizers[:5]
longest_median = sorted_tokenizers[-5:]

fig = go.Figure()
for name, metrics in shortest_median + longest_median:
    fig.add_trace(go.Bar(x=[name], y=[metrics["Median"]], name=name))
fig.update_layout(
    title="Top Tokenizers by Shortest and Longest Median Token Length",
    xaxis_title="Tokenizer",
    yaxis_title="Median Token Length",
)
st.plotly_chart(fig)
