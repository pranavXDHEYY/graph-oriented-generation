###### Note: I am currently seeking an arXiv endorser in the cs.IR and cs.AI category for the formal preprint of this paper. If you are eligible and find this work valuable, please reach out or endorse directly at https://urldefense.com/v3/__https://arxiv.org/auth/endorse?x=OVESPR__;!!DaRZpAeNFA!bDQ8GlkoWQn5HCz0RtmrPvpR_l4miMk56L2WuvsMq0eBQiWcGhq05BYb-bQV0b13Ewtg7RMYyl0fmLttsZM$!

---

# GOG Benchmark (Graph-Oriented Generation)

This repository evaluates the efficiency of **Symbolic Reasoning Model (SRM)** context isolation (GOG) compared to standard **Retrieval-Augmented Generation (RAG)** for large codebase understanding.

## Architecture

*   **Python Engine:** Orchestrates the benchmark, parses the codebase, and interacts with the LLM API.
*   **SRM Engine:** Uses `networkx` to build a dependency graph of the codebase and isolate relevant files for a given prompt.
*   **Benchmark Harness:** A/B tests the context load and execution time between a full codebase dump (RAG) and isolated context (GOG).

## 🛑 Current Status

**Active Research Prototype — Contributions Welcome**

> **Update:** Thank you for the incredible response on Hacker News and Reddit!  
> The project reached **20+ stars and multiple forks within the first 24 hours.**

GOG is currently an **active research prototype** (Paper #2 in progress).  
My limited development time is focused on advancing the **core mathematical engine**, specifically:

- **$O(1)$ plasticity**
- **deterministic traversal**

If you're interested in helping build the surrounding ecosystem, **community contributions are highly encouraged.**

If the idea of helping **challenge traditional Vector RAG architectures** sounds interesting, take a look at the open issues.

---

## 🗺️ Roadmap & Areas for Contribution

Below are several areas where contributions would have a meaningful impact.

### 🌍 Language Expansion

The **SRM AST parser** is currently optimized for:

- Python
- TypeScript

Additional language support would be valuable for:

- Go
- Rust
- Java

---

### 🧪 Model Benchmarking

The current benchmark suite focuses on **Qwen 0.8B**.

Next steps include expanding the benchmark gauntlet to evaluate:

- **Llama 3 (8B)**
- **Mistral variants**
- additional lightweight models

---

### 🖥 CLI & Benchmark Output

Improving the **terminal output and visualization** of benchmark results, particularly:

- clearer **$O(1)$ speed comparisons**
- more intuitive **token reduction metrics**
- improved CLI readability for rapid experimentation

---

If you're interested in contributing, feel free to:

- open a PR
- comment on an issue
- suggest improvements to the roadmap

All help is appreciated!

---

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install OpenCode CLI:**
    The benchmarking suite uses the `opencode` CLI for all LLM interactions. Install it via NPM:
    ```bash
    npm install -g opencode
    ```

3.  **Generate the Maze:**
    Inflate the target repository with 50+ dummy files and a hidden "needle" component.
    ```bash
    python3 generate_dummy_repo.py
    ```

## Running the Benchmark

There are two primary ways to run the benchmark: via the Cloud-based OpenCode CLI or purely locally using an open-source Small Language Model (SLM) via Ollama.

### 1. Cloud Execution (OpenCode CLI)
Use this method to benchmark performance using state-of-the-art cloud models.

```bash
python3 benchmark_cloud_cli.py
```

### 2. Local SLM Execution (Ollama)
Use this method to prove that GOG is so efficient that it can run entirely on local resources using small models like `qwen`. This removes API latency and costs completely.

**Install Ollama & Prepare the Model:**
1. Download mapping and install Ollama from [ollama.com](https://ollama.com) or run:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the specified local LLM (e.g. `qwen3.5:0.8b` or whichever you prefer):
   ```bash
   ollama pull qwen3.5:0.8b
   ```
3. Run the local benchmark:
   ```bash
   python3 benchmark_local_llm.py
   ```

## Expected Results

The SRM Engine should demonstrate a **70%+ reduction in token usage on average** by deterministically tracing the precise dependency paths, ignoring the dozens of noise components that plague typical Vector RAG setups. Furthermore, the Local Compute Time metric will highlight the fundamental difference in overhead between $O(n)$ vector scaling and $O(1)$ graph traversal.
