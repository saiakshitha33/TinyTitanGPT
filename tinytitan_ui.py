import os
import torch

# Suppress Streamlit-PyTorch watcher error
torch.classes.__path__ = []

import torch.nn as nn
import streamlit as st
from tinytitan_gpt import TinyTitanGPT, generate_note, build_vocab

# === Page Config ===
st.set_page_config(
    page_title="TinyTitanGPT",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Dark Theme Styling ===
st.markdown(
    """
    <style>
        .stApp {
            background-color: #121212;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        label, .stTextInput > label, .stTextArea > label, .stSelectbox > label, .stSlider > label {
            color: white !important;
            font-weight: 600;
        }
        .stTextArea textarea {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .stCheckbox {
            color: white !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stDownloadButton>button {
            background-color: #2196F3;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        code {
            background-color: #2e2e2e;
            color: #f8f8f2;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# === Safe Checkpoint Loader ===
def safe_load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    for key in checkpoint:
        if key in model_state and checkpoint[key].shape != model_state[key].shape:
            print(f"⚠️ Skipping '{key}' due to shape mismatch: "
                  f"{checkpoint[key].shape} vs {model_state[key].shape}")
            checkpoint[key] = model_state[key]
    model.load_state_dict(checkpoint, strict=False)
    print("✅ Model loaded (with safe fallback)")

# === Load Model + Vocab ===
def load_model_and_vocab():
    if not os.path.exists("input.txt"):
        st.error("❌ input.txt not found. Please add your training data.")
        st.stop()

    with open("input.txt", "r") as f:
        text = f.read()

    chars, stoi, itos, encode, decode = build_vocab(text)
    vocab_size = len(chars)

    model = TinyTitanGPT(vocab_size).to("cpu")
    checkpoint_path = "tinytitan_checkpoint.pt"

    if os.path.exists(checkpoint_path):
        safe_load_checkpoint(model, checkpoint_path)
        model.eval()
    else:
        st.warning("⚠️ No checkpoint found. Please train TinyTitan first.")

    return model, stoi, decode

model, stoi, decode = load_model_and_vocab()

# === Session State for Prompt History ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Prompt Cleaning Helper ===
def extract_first_output_section(result):
    """Keep content after first ### Output: and cut if another ### Input appears."""
    if "### Output:" in result:
        section = result.split("### Output:")[1]
        if "### Input:" in section:
            section = section.split("### Input:")[0]
        return "### Output:\n" + section.strip()
    return result.strip()

# === Tabs for UI Layout ===
tab1, tab2 = st.tabs(["📝 Generate Note", "⚙️ Settings"])

# === Prompt Input ===
with tab1:
    example = st.selectbox("🧪 Example Prompts", [
        "Pt. complains of headache and dizziness.",
        "Patient reports abdominal pain and nausea.",
        "Sore throat for the past 3 days.",
        "Experiencing chest pain during exertion.",
        "History of high blood pressure and blurred vision."
    ])
    prompt = st.text_area("🩺 Enter Clinical Prompt", example)
    clear_after = st.checkbox("🧹 Clear prompt after generation", value=False)

# === Generation Controls ===
with tab2:
    tokens = st.slider("🧠 Max Tokens", 50, 1000, 400)
    temperature = st.slider("🎯 Temperature", 0.5, 1.5, 1.0)
    top_k = st.slider("🔢 Top-k Sampling", 5, len(stoi), min(30, len(stoi)))

# === Generate Button Logic ===
if st.button("🚀 Generate SOAP Note"):
    with st.spinner("🧠 TinyTitan is generating..."):
        if prompt.strip().startswith("### Input:"):
            formatted_prompt = prompt.strip()
        else:
            formatted_prompt = f"### Input:\n{prompt.strip()}\n\n### Output:\n"

        raw_result = generate_note(formatted_prompt, model, stoi, decode, tokens, temperature, top_k)
        result_clean = extract_first_output_section(raw_result)

        st.session_state.history.append({
            "prompt": prompt.strip(),
            "output": result_clean
        })

        st.markdown("---")
        st.markdown("### 🧾 Prompt Used")
        st.code(formatted_prompt, language="markdown")

        st.markdown("### ✨ Generated SOAP Note")
        st.code(result_clean, language="markdown")
        st.success("✅ Note generated successfully!")
        st.download_button("💾 Download Note", result_clean, file_name="tinytitan_note.txt")

        if clear_after:
            prompt = ""

# === Prompt History Sidebar ===
with st.sidebar:
    st.markdown("## 🕓 History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"**{i}. Prompt:**")
            st.code(item['prompt'], language="markdown")
            st.markdown("**Output:**")
            st.code(item['output'], language="markdown")
            st.markdown("---")
    else:
        st.info("No history yet. Generate something!")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>"
    "Made with ❤️ by <b>Akshitha Kota</b><br>"
    "<a href='https://github.com/saiakshitha33' target='_blank'>GitHub</a> | "
    "<a href='https://huggingface.co' target='_blank'>Model Info</a>"
    "</div>",
    unsafe_allow_html=True
)
