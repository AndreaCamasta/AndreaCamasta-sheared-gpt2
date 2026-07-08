# Sheared GPT-2: Reproducing Structured Pruning and Dynamic Batch Loading from Sheared LLaMA

A reproduction and adaptation of the **Sheared LLaMA** pruning pipeline on **GPT-2**, developed as the final project for the **Natural Language Processing** course.

This project implements the core ideas presented in:

> Xia et al., **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning**, 2023.

Instead of reproducing the paper on a multi-billion-parameter LLaMA model, this project adapts the proposed methodology to the much smaller **GPT-2** architecture while preserving the overall pruning and retraining pipeline.

---

# Project Overview

Large Language Models are expensive to train and deploy due to their computational and memory requirements.

Sheared LLaMA proposes an alternative to traditional model compression:

1. **Learn a smaller architecture through structured pruning**
2. **Materialise the resulting dense model**
3. **Recover performance using Dynamic Batch Loading (DBL)**

This repository reproduces these ideas on GPT-2 by implementing:

- structured layer pruning
- structured FFN pruning
- learnable pruning masks
- constrained optimisation toward a target architecture
- dense model reconstruction
- Dynamic Batch Loading based on reference-model loss ratios
- evaluation on both in-domain and out-of-domain datasets

Although simplified in several aspects (see Limitations below), the implementation successfully reproduces the qualitative behaviour described in the original paper.

---

# Repository Structure

```text
AndreaCamasta-sheared-gpt2/
│
├── models/
│   ├── __init__.py
│   ├── masked_gpt2.py
│   └── build_sheared_gpt2_structured.py
│
├── pruning/
│   ├── __init__.py
│   ├── masks.py
│   └── objective.py
│
├── training/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── dynamic_data.py
│   ├── dynamic_trainer.py
│   ├── eval_utils.py
│   └── pruning_trainer.py
│
├── sheared_gpt2_layers_ffn_10k_dynamic/
│
├── Sheared_GPT2_Demo.ipynb
│
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Methodology

The implementation follows the same high-level pipeline as the paper.

```
                Pretrained GPT-2
                      │
                      ▼
         Learnable Structured Masks
      (Layers + Feed-Forward Dimensions)
                      │
                      ▼
        Constrained Pruning Optimisation
                      │
                      ▼
          Materialise Dense GPT-2
          (Reduced Architecture)
                      │
                      ▼
       Dynamic Batch Loading Training
        (Reference-model loss ratios)
                      │
                      ▼
             Final Sheared GPT-2
```

---

# Implemented Components

## 1. Structured Pruning

Learnable masks are introduced for:

- Transformer layers
- Feed-forward hidden dimensions

The pruning objective jointly optimises:

- language modelling loss
- architecture constraints

To match a predefined target architecture.

Unlike unstructured pruning, this produces a **dense and deployable model**.

---

## 2. Model Materialization

After pruning, the selected layers and FFN dimensions are physically copied into a new GPT-2 architecture.

The resulting model:

- contains fewer parameters
- requires no pruning masks
- can be deployed as a standard Hugging Face model

---

## 3. Dynamic Batch Loading

The paper proposes Dynamic Batch Loading (DBL) to recover the performance lost during pruning.

This repository implements the same idea.

During fine-tuning:

- A frozen reference GPT-2 computes the language-model loss
- The ratio between student and teacher losses is maintained for each domain
- Sampling probabilities are dynamically updated to focus training on domains where the compressed model underperforms

The implemented domains are:

- WikiText
- Yelp Reviews
- AG News

---

## 4. Evaluation

The models are evaluated using:

- parameter count
- inference speed
- perplexity
- qualitative text generation

Performance is also measured on unseen datasets to study the specialisation induced by Dynamic Batch Loading.

---

# Results

The experiments show the same qualitative behaviour reported in the paper.

- Significant reduction in model size
- Faster inference
- Performance degradation immediately after pruning
- Strong recovery after Dynamic Batch Loading
- Improved performance on target domains
- Reduced generalisation on unrelated domains

These observations closely match the conclusions of Sheared LLaMA despite the much smaller model scale.

---

# Differences from the Original Paper

For computational reasons, several simplifications were introduced.

### Original Paper

- LLaMA (billions of parameters)
- Gumbel-Softmax stochastic pruning gates
- Layer, attention-head and FFN pruning
- Large multi-domain corpus
- Massive pre-training compute

### This Project

- GPT-2 (124M parameters)
- Deterministic sigmoid masks
- Layer and FFN pruning
- Three representative domains
- Google Colab implementation

The goal was not an exact reproduction, but rather to validate the underlying methodology at a smaller scale.

---

# Main Findings

- Structured pruning successfully compresses GPT-2 while preserving a dense architecture.
- Dynamic Batch Loading is essential to recover performance after pruning.
- The compressed model becomes increasingly specialised toward the selected training domains.
- The qualitative behaviour closely follows the findings reported in the original Sheared LLaMA paper.

---

# Requirements

Main libraries:

- Python 3.11+
- PyTorch
- Transformers
- Datasets
- Accelerate
- NumPy

Users can use:
```
pip install -r requirements.txt
```
To directly install the required libraries

---

# Reference

Xia, M., et al.

**Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.**

arXiv:2310.06694 (2023)

https://arxiv.org/abs/2310.06694

---

# Acknowledgements

This repository was developed as the final project for the **Natural Language Processing** course.

Its purpose is educational and research-oriented: to study, reproduce, and experimentally validate the structured pruning methodology proposed in the Sheared LLaMA paper on a smaller GPT-2 model.
