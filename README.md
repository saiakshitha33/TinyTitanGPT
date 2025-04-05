
# TinyTitanGPT

TinyTitanGPT is a custom, lightweight GPT-style transformer model built from scratch using PyTorch. This project was developed to explore, implement, and analyze the architectural components of transformer-based language models and their behavior in real-world text generation tasks.

The model is trained on a character-level dataset and wrapped with a responsive Streamlit-based user interface to allow real-time generation of SOAP-format clinical notes from short prompts.

---

## Project Objectives

- Implement a GPT-style transformer from scratch using PyTorch
- Gain a deep understanding of self-attention, positional embeddings, and layer normalization
- Experiment with vocabulary construction, tokenization, and sampling strategies (temperature and top-k)
- Build a user-friendly interface to demonstrate practical use cases of autoregressive text generation
- Apply the model to structured clinical note generation in a SOAP (Subjective, Objective, Assessment, Plan) format

---

## Features

- Multi-layer Transformer architecture (6 blocks, 8 attention heads)
- Character-level encoding and decoding
- Configurable sampling parameters: temperature and top-k
- Streamlit-based interactive UI with:
  - Dark mode styling
  - Prompt templates
  - Prompt history saving
  - Output download as `.txt`
- Lightweight enough for fast experimentation and deployment on Streamlit Cloud

---


## How to Run Locally
'''
1. Clone the repository
  bash
git clone https://github.com/<your-username>/TinyTitanGPT.git
cd TinyTitanGPT


2.Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows


3.Install dependencies
bash
pip install -r requirements.txt

4.Run the app

bash
streamlit run tinytitan_ui.py

'''

---

## 🧠 Technical Overview (In-Depth)

| **Component**     | **Detail** |
|-------------------|------------|
| **Language Model** | TinyTitanGPT is a custom-built, transformer-based language model inspired by GPT (Generative Pretrained Transformer). It learns to predict the next character in a sequence, enabling it to generate structured clinical text from scratch. |
| **Framework** | Implemented using **PyTorch**, a dynamic deep learning framework that provides full control over tensor operations, backpropagation, and model design — ideal for building neural networks from the ground up. |
| **Layers** | The model consists of **6 stacked transformer blocks**, each containing self-attention and feedforward layers. These layers allow the model to learn hierarchical patterns and dependencies across different parts of the text input. |
| **Attention** | Each block uses **8 attention heads**, enabling the model to attend to multiple parts of the input sequence simultaneously. Multi-head attention allows the network to understand different types of relationships between characters (e.g., temporal, grammatical, semantic). |
| **Embeddings** | Each input character is mapped to a **128-dimensional vector**. This embedding layer helps the model learn distributed representations of tokens, making it easier to capture similarities and patterns in the training data. |
| **Training Data** | The model is trained on **character-level** sequences. Instead of using full words or tokens, it processes individual characters, which is useful for smaller vocabularies and allows learning of fine-grained syntax like clinical abbreviations. |
| **Sampling** | During text generation, **temperature** and **top-k** sampling strategies are used to control creativity and coherence. Temperature controls randomness (lower = more deterministic), and top-k restricts sampling to the top-k most probable next characters. |
| **Checkpoints** | The model saves its weights to disk every **500 training steps**. This allows users to resume training from the last checkpoint without losing progress, and to experiment with different generation techniques using the latest learned state. |

---


🔮 Future Roadmap
 
 Expand dataset with real or synthetic SOAP notes


👩‍⚕️ Created By
Sai Akshitha Reddy Kota

Experienced Python Developer | Data Scientist |  AI

🔗 GitHub: [https://github.com/saiakshitha33]

📫 [saiakshitha.kota@gmail.com] 

⚠️ Disclaimer
This project is for educational and research purposes only. It is not intended for clinical use or to replace licensed medical professionals.
