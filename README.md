# MAC-SLU: A Benchmark for Multi-Intent Spoken Language Understanding in Automotive Cabins

<p align="center">
    <a href="https://arxiv.org/abs/2512.01603"><img src="https://img.shields.io/badge/arXiv-Paper-brightgreen" alt="arXiv Paper"></a>
    <a href="https://huggingface.co/datasets/Gatsby1984/MAC_SLU"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow" alt="HuggingFace Dataset"></a>
    <a href="https://github.com/Gatsby-web/MAC_SLU"><img src="https://img.shields.io/badge/GitHub-Repo-lightgreen" alt="GitHub Repo"></a>
</p>

---

[](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)
[](https://www.python.org/downloads/release/python-3100/)

This repository contains the code and resources for **MAC-SLU**, a benchmark designed to evaluate Spoken Language Understanding systems on complex, multi-intent user commands within an automotive environment.

## 🚀 Getting Started

### 1\. Download the Dataset

The complete MAC-SLU dataset is hosted on the Hugging Face Hub.

  * **Dataset Link:** [Gatsby1984/MAC\_SLU](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)

### 2\. Prepare the Environment

Our experiments are divided into two main approaches: **In-Context Learning (ICL)** and **Supervised Fine-Tuning (SFT)**. Please set up the appropriate environment for the method you wish to use.

## 🛠️ Usage

### In-Context Learning (ICL)

#### Environment Setup

Our ICL code relies on `vLLM`. The required version depends on the model you are using. All experiments were conducted with **Python 3.10**.

  * For **Qwen3** experiments: `pip install vllm==0.9.2`
  * For **Qwen2.5-Omni** experiments: `pip install vllm==0.8.5.post1`

#### Running ICL Experiments

**Step 1: Deploy the Model with vLLM**
(This step is not required if you are using a commercial API.)

Open a terminal and run the following command to start the vLLM server. This example is for `Qwen2.5-Omni-7B`.

```bash

export CUDA_VISIBLE_DEVICES=0

vllm serve /path/to/your/Qwen2.5-Omni-7B \
  --served-model-name Qwen2.5-Omni-7B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 12355 \
  --uvicorn-log-level warning \
  --disable-log-requests \
  --max-model-len 32768
```

**Step 2: Run Inference**

Once the server is running, open a new terminal and execute the inference script.

```bash
python slu_icl.py \
    --provider local \
    --input-file /path/to/test_set.jsonl \
    --audio-dir /path/to/audio_test_directory \
    --output-file /path/to/prediction.jsonl \
    --model-name Qwen2.5-Omni-7B \
    --api-base http://0.0.0.0:12355/v1
```

  * **Note:** For other models, you may need to change `--model-name` and the model path in the `vllm serve` command. To use a commercial API, change `--provider` to the appropriate name and configure the necessary API keys.

**Step 3: Evaluation**

```bash
python metrics.py prediction.jsonl icl_label.jsonl
```
-----

### Supervised Fine-Tuning (SFT)

#### Environment Setup

For SFT experiments, we use the efficient **LLaMA-Factory** framework. Please follow the official instructions to install and set up the environment.

  * **Framework:** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

#### Training Instructions

We recommend using a **LoRA-SFT** approach for fine-tuning.

1.  **Prepare your dataset** using the format required by LLaMA-Factory.
2.  **Configure your training run** by selecting a model, dataset, and setting the LoRA hyperparameters.

