# GEN-AI
---

````markdown
# 🧠 Text Completion Generator

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-GPT2-orange)
![Transformers](https://img.shields.io/badge/🤗-Transformers-ff69b4)

A terminal-based AI-powered **text completion tool** that leverages Hugging Face Transformers and PyTorch to generate coherent and creative continuations of input prompts. Comes with both an interactive shell and CLI support for flexible usage.

---

## ✨ Features

- ✅ Easy-to-use command-line interface  
- ✅ Interactive mode with real-time settings adjustment  
- ✅ Supports any Hugging Face causal language model (e.g., `gpt2`, `EleutherAI/gpt-neo`)  
- ✅ Fine-tuned parameters: `temperature`, `top_k`, `top_p`, etc.  
- ✅ Rich terminal UI via the `rich` library  
- ✅ GPU acceleration with CUDA (if available)  

---

## 📦 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/text-completion-generator.git
cd text-completion-generator
pip install -r requirements.txt
````

### Requirements

* Python 3.7+
* PyTorch (>=1.9.0)
* Transformers (>=4.18.0)
* Rich (>=12.0.0)

---

## 🚀 Usage

### 🔁 Interactive Mode

```bash
python 1.py --interactive
```

* Type any prompt to generate completions.
* Adjust settings like `temperature` or `max_length` by typing `settings`.
* Exit with `exit` or `quit`.

### 💬 Single-Prompt Mode

```bash
python 1.py --prompt "Once upon a time" --max-length 100 --temperature 0.7 --num-sequences 2
```

### 💻 Force CPU/GPU

```bash
python 1.py --prompt "The future of AI is" --device cpu
```

---

## ⚙️ Command-Line Arguments

| Flag              | Description                                       | Default |
| ----------------- | ------------------------------------------------- | ------- |
| `--model`         | Hugging Face model name (e.g., `gpt2`, `gpt-neo`) | `gpt2`  |
| `--prompt`        | Prompt text for one-shot completion               | —       |
| `--max-length`    | Maximum length of generated text                  | `50`    |
| `--temperature`   | Controls creativity/randomness of output          | `0.8`   |
| `--top-k`         | Top-k sampling cutoff                             | `50`    |
| `--top-p`         | Nucleus sampling probability threshold            | `0.9`   |
| `--num-sequences` | Number of completion outputs                      | `1`     |
| `--interactive`   | Enable interactive CLI mode                       | `False` |
| `--device`        | `cpu` or `cuda`; auto-detect if not set           | Auto    |

---

## 🧪 Example Output

```
Enter text to complete: The secret to happiness is

Completions:
────────────────────────────────────────────────
The secret to happiness is accepting who you are and finding peace in the present moment.
────────────────────────────────────────────────
```

---

## 🛠 Built With

* [🤗 Transformers](https://huggingface.co/transformers/) — Model loading & text generation
* [🧠 PyTorch](https://pytorch.org/) — Model inference engine
* [🎨 Rich](https://github.com/Textualize/rich) — Beautiful CLI output

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Thanks to the Hugging Face team and the open-source community for building accessible NLP tools.

```
