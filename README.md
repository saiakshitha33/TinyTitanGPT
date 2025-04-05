# TinyTitanGPT - Medical Note Generator

TinyTitanGPT is a mini transformer-based language model trained to generate structured clinical notes (in SOAP format) from short, natural-language medical prompts.

Built entirely from scratch in PyTorch, it demonstrates how transformer-based architectures can be adapted for real-world healthcare applications such as clinical documentation automation.

---

## Features

- ✅ Implemented in a single file: `tinytitan_gpt.py`
- ✅ Transformer architecture with multi-head self-attention
- ✅ Learns from prompt → SOAP note examples
- ✅ Supports temperature + top-k sampling for creative control
- ✅ Model checkpointing every 500 steps
- ✅ Includes CLI-based generation (no UI required)

---

## Example Use

**Prompt:**
Pt. complains of chest tightness and shortness of breath during exercise.


**Generated Output:**
Output:
Subjective: 45-year-old male presents with exertional chest tightness and dyspnea. Denies palpitations or dizziness.

Objective: BP: 138/85 mmHg, HR: 88 bpm. Lungs clear to auscultation. No lower extremity edema.

Assessment: Likely stable angina.

Plan:

Order ECG and exercise stress test

Start aspirin 81mg daily

Lifestyle modifications

Follow-up in 1 week



---
Just run the training script :

python tinytitan_gpt.py


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
