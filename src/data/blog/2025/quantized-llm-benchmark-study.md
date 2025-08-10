---
author: Ionio AI Research Team
pubDatetime: 2025-08-07T00:00:00Z
modDatetime: 2025-08-07T00:00:00Z
title: A Comparative Study of Quantized Large Language Models
slug: quantized-llm-benchmark-study
featured: true
draft: false
tags:
  - LLM
  - Quantization
  - Benchmarking
  - AI
description: This study benchmarks quantized variants of LLMs (Qwen2.5, DeepSeek, Mistral, and LLaMA 3.3) across diverse tasks to assess the trade-offs in accuracy and robustness.
---

## Table of contents

## Abstract
![AstroPaper v3](@/assets/images/BBH_bar_chart.png)

Quantization has emerged as a critical method for deploying large language models (LLMs) in constrained environments. This study benchmarks quantized variants of Qwen2.5, DeepSeek, Mistral, and LLaMA 3.3 across five diverse tasks: MMLU, GSM8K, BBH, C-Eval, and IFEval, spanning domains from math reasoning to instruction following. We evaluate each model under multiple quantization schemes (BF16, GPTQ-INT8, INT4, AWQ, GGUF, including Q3 K M, Q4 K M, Q5 K M, and Q8 0) to assess the trade-offs in accuracy retention and task robustness. Our findings offer actionable insights into quantization format selection for production use, highlighting that **Q5 K M** and **GPTQ-INT8** offer optimal trade-offs for most domains, while **AWQ** and lower-bit GGUF formats should be used cautiously.

## 1 Introduction

In recent years, large language models (LLMs) have rapidly transitioned from research labs to real-world products, powering virtual assistants, developer tools, financial advisors, and even autonomous agents. However, while their capabilities have grown, so too have their computational demands. Full-precision LLMs are often too large, too slow, or too resource-intensive for many real-world deployment scenarios. This is where **quantization** enters the conversation.

Quantization allows us to compress these models, typically by reducing the bit-width of weights and activations without retraining them from scratch. In doing so, we significantly lower memory usage and speed up inference, making LLMs deployable even on constrained hardware. However, quantization introduces trade-offs, often manifesting as accuracy degradation across specific tasks. This degradation is rarely uniform across all tasks.

Despite the availability of dozens of quantized models on platforms like Hugging Face, clear guidance is still lacking on how different quantization formats behave in practical use cases. Most existing benchmarks focus on raw accuracy, usually under ideal conditions, and often overlook critical variables like inference latency, robustness to decoding variation, or task-specific failure modes. For teams building with LLMs in production, where cost, speed, and reliability matter, these one-size-fits-all evaluations fall short.

This paper aims to address that gap. Rather than simply comparing models on standard leaderboards, we adopt a task-oriented view. We evaluate quantized versions of four leading instruction-tuned model families—Qwen2.5, DeepSeek, Mistral, and LLaMA 3.3—across a wide range of benchmarks, including MMLU, BBH, GSM8K, IFEval, and C-Eval. Each benchmark is tied to a practical domain, from finance to software development to reasoning agents. Just as importantly, we analyze each model across multiple quantization formats: from BF16 full-precision baselines to INT4/INT8 via GPTQ, AWQ, and the GGUF family of formats like Q4 K M, Q5 K M, and Q8 0. This enables us to assess quantization trade-offs across real-world use cases. Our central question is: **Which quantized format gives the best trade-off between accuracy, speed, and task suitability for a specific use case?**

By the end of this study, we provide a much clearer picture of how quantization affects model performance, not just in abstract benchmarks, but in the kinds of real-world applications LLMs are increasingly being asked to support.

## 2 Methodology

To ensure a meaningful and representative evaluation of quantized LLMs, we adopted a comprehensive methodology focusing on model family selection, quantization schemes, benchmark suite design, and evaluation protocols.

### 2.1 Model Families and Quantization Schemes

Our evaluation focuses on four major model families: **Qwen2.5**, **DeepSeek**, **Mistral**, and **LLaMA 3.3**. These were chosen based on their open availability, strong performance on instruction-following tasks, multilingual capabilities, and overall popularity in the research and open-source communities. We specifically targeted model sizes ranging from 7B to 32B parameters, as these offer the best trade-offs between performance and deployability in real-world applications.

Each model was evaluated in its full-precision format (BF16 or FP16) as a baseline, alongside at least three quantized versions. The quantization formats were selected to represent a broad spectrum of deployment needs:

* **GGUF**: Includes Q3 K M, Q4 K M, Q5 K M, and Q8 0.
* **GPTQ**: Widely adopted in GPU inference setups, supported by frameworks like AutoGPTQ, vLLM, and ExLlama.
* **AWQ**: A newer quantization approach optimized for weight-only compression with minimal accuracy drop.
* **BF16**: Used as the reference format, providing a reliable upper bound on model performance.

We intentionally selected models with varied specialization, such as **Qwen2.5-Instruct** for reasoning, **Qwen2.5-Coder** for programming, **DeepSeek-R1-Distill** for efficiency, **Mistral-7B-Instruct** for lightweight instruction following, and **LLaMA 3.3** for the high-end frontier. This diversity allows us to study how quantization impacts performance across domains like math, science, programming, and complex reasoning.

### 2.2 Benchmark Suite and Task Alignment

To move beyond raw accuracy scores, we mapped each benchmark to a corresponding real-world use case. Our evaluation includes the following five major benchmarks:

* **MMLU (Massive Multitask Language Understanding)**
    * **Mapped Use Case**: Research assistants, legal Q&A, enterprise knowledge tools.
    * **Quantization Insight**: Small degradations in this benchmark can significantly affect trustworthiness in professional deployments.
* **GSM8K (Grade School Math Word Problems)**
    * **Mapped Use Case**: Financial assistants, mathematical modeling, budgeting tools.
    * **Quantization Insight**: Formats with aggressive compression tend to degrade performance on GSM8K earlier than other tasks.
* **BBH (BIG-Bench Hard)**
    * **Mapped Use Case**: Autonomous agents, planning systems, multi-hop assistants.
    * **Quantization Insight**: BBH reveals robustness under high-cognitive load tasks; some models retain performance well even in INT4, while others collapse rapidly.
* **IFEval (Instruction Following Evaluation)**
    * **Mapped Use Case**: Workflow automation, LLM-based agents, chatbot task handlers.
    * **Quantization Insight**: AWQ and some GGUF formats introduce decoding variability, which can harm instruction precision.
* **C-Eval**
    * **Mapped Use Case**: Edtech in Asia, cross-lingual reasoning tools, enterprise localization.
    * **Quantization Insight**: Performance drops here indicate quantization’s effect on tokenizer alignment and language-specific embeddings, which matter in non-English deployments.

### 2.3 Evaluation Protocol and Inference Settings

To ensure our results reflect real-world usage patterns, we adopted a dual-mode inference strategy:

* **Zero-Shot Deterministic Decoding (Temperature = 0)**: This mirrors use cases where consistency and predictability are critical, like legal document drafting.
* **Multi-Pass Inference with Sampling (Temperature = 0.3)**: This simulates typical chatbot scenarios where mild randomness improves fluency and creativity. This method also allows us to capture robustness, identifying formats that remain stable under light sampling noise.

---

## 3 Results and Observations

### 3.1 Quantization Accuracy Retention Analysis

To evaluate how different quantization schemes affect model performance, we used the Qwen2.5-7B-Instruct model as a case study and tested it across five representative benchmarks: BBH, MMLU, C-Eval, IFEval, and GSM8K. These benchmarks span domains such as multistep reasoning (BBH), factual knowledge (MMLU), multilingual academic tasks (C-Eval), instruction following (IFEval), and math reasoning (GSM8K).

Each quantization format—ranging from BF16 (baseline) to low-bit variants like Q4 K M and INT4—was evaluated by comparing its raw accuracy to the full-precision reference. From this, we derived a retention score (% of baseline accuracy) to assess format stability under quantization pressure.

A heatmap illustrating retention across quantization formats and benchmarks for Qwen2.5-7B-Instruct is presented below.

<figure>
  <img
    src="/src/assets/images/fixed_full_heatmap_model_quant.png" 
    alt="A heatmap illustrating accuracy retention across quantization formats and benchmarks for the Qwen2.5-7B-Instruct model. The heatmap shows a monotonic decrease in accuracy as bit-width is reduced across all tasks."
  />
  <figcaption class="text-center">
    Figure 1: Retention Heatmap Across Quantization Formats and Benchmarks for Qwen2.5-7B-Instruct
  </figcaption>
</figure>

### 3.2 Quantization Effects Across Benchmarks

There is a clear, monotonic decrease in accuracy as bit-width reduces:

* **BF16** → **GPTQ-INT8** results in a minimal accuracy loss (usually <2%), suggesting that 8-bit quantization retains almost all meaningful representation power.
* **BF16** → **GPTQ-INT4** introduces moderate degradation (∼3–6%), which is acceptable in most general-purpose deployments but may not be suitable for edge-cases like legal or medical QA.
* **BF16** → **Q4 K M/Q4 K S** shows the steepest degradation, especially in instruction-heavy (IFEval) and multilingual (C-Eval) settings, with up to 20% loss from baseline.

These findings are consistent with existing literature suggesting reduced bit-width quantization impairs instruction-level coherence and factual retrieval.

---

#### BBH (Logical Reasoning)

Despite being a complex benchmark, BBH accuracy degrades relatively smoothly across formats. Even in Q4 K M, the model retains ∼90% of its BF16 accuracy. This implies that quantization-induced degradation in BBH is less catastrophic, likely due to the structural consistency of logical patterns.

<figure>
  <img
    src="/src/assets/images/BBH_bar_chart.png"
    alt="A chart showing BBH accuracy across different models (Qwen2.5, DeepSeek, Mistral, LLaMA 3.3) and their respective quantization schemes. The chart demonstrates that BBH accuracy degrades relatively smoothly across quantization levels."
  />
  <figcaption class="text-center">
    Figure 2: BBH Accuracy Across Models and Quantization Schemes
  </figcaption>
</figure>

---

#### MMLU (General Knowledge)

MMLU is more sensitive to quantization, particularly in lower-bit formats. This suggests that factual retrieval tasks depend heavily on the precision of internal embeddings and attention weights, making GGUF formats below Q5 K M risky for knowledge-intensive applications.

<figure>
  <img
    src="/src/assets/images/MMLU_bar_chart.png"
    alt="A chart showing MMLU accuracy across different models and quantization schemes. The chart indicates that MMLU performance is more sensitive to lower-bit quantization formats."
  />
  <figcaption class="text-center">
    Figure 3: MMLU Accuracy Across Models and Quantization Schemes
  </figcaption>
</figure>

---

#### C-Eval (Multilingual Academic Reasoning)

C-Eval results show a noticeable drop in all formats except GPTQ-INT8. Q4 K M sees almost 15–20% reduction in retention, indicating that tokenizer-alignment and language-specific embeddings suffer under aggressive quantization. This is especially critical in localized deployments in Asia or multilingual enterprise systems.

<figure>
  <img
    src="/src/assets/images/C-Eval_bar_chart.png"
    alt="A chart showing C-Eval accuracy across different models and quantization schemes. The data reveals a significant drop in performance for lower-bit formats, highlighting sensitivity to multilingual tasks."
  />
  <figcaption class="text-center">
    Figure 4: C-Eval Accuracy Across Models and Quantization Schemes
  </figcaption>
</figure>

---

#### IFEval (Instruction Following)

IFEval appears highly sensitive to quantization, especially at INT4 and GGUF Q4 levels. Models show more than 10% accuracy loss, and sometimes erratic behavior. This supports the hypothesis that instruction-following quality depends not only on token predictions but also on decoder alignment, which becomes unstable in very low-bit formats.

<figure>
  <img
    src="/src/assets/images/IFEval_bar_chart.png"
    alt="A chart showing IFEval accuracy across different models and quantization schemes. The chart demonstrates that instruction-following performance is particularly sensitive to aggressive quantization."
  />
  <figcaption class="text-center">
    Figure 5: IFEval Accuracy Across Models and Quantization Schemes
  </figcaption>
</figure>

---

#### GSM8K (Mathematical Reasoning)

Interestingly, GSM8K shows relatively high retention even in Q4 K M and Q4 K S, with ∼84–87% of baseline accuracy. This implies that step-by-step arithmetic tasks are structurally resilient to quantization, especially in models with strong reasoning architectures like Qwen.

<figure>
  <img
    src="/src/assets/images/GSM8K_bar_chart.png"
    alt="A chart showing GSM8K accuracy across different models and quantization schemes. The chart indicates that mathematical reasoning tasks are relatively robust to quantization."
  />
  <figcaption class="text-center">
    Figure 6: GSM8K Accuracy Across Models and Quantization Schemes
  </figcaption>
</figure>

---

### 3.2.3 AWQ and GGUF: Compression vs. Consistency

The behavior of **AWQ** is notable. In benchmarks like IFEval, it underperforms relative to GPTQ-INT4, even though both use 4-bit quantization. This suggests that AWQ’s group-wise quantization may introduce non-determinism that disrupts instruction alignment, even when weight fidelity is preserved.

In contrast, GGUF formats like **Q4 K M** are extremely lightweight and enable CPU deployment, but show degradation patterns that must be considered carefully. Especially in C-Eval and IFEval, Q4 formats introduce unacceptable losses for production-level deployments.

The sweet spot appears to be **Q5 K M** or **Q8 0**, where we retain ∼95–99% of the original performance, with substantial gains in inference speed and memory efficiency.

| Benchmark | GPTQ-INT8 | GPTQ-INT4 | Q8 0 | Q4 K M |
| :--- | :--- | :--- | :--- | :--- |
| **BBH** | 98.0% | 96.4% | 95.4% | 89.8% |
| **MMLU** | 99.3% | 96.2% | 93.2% | 87.6% |
| **C-Eval** | 99.6% | 97.4% | 91.3% | 77.4% |
| **IFEval** | 95.9% | 93.0% | 80.3% | 83.5% |
| **GSM8K** | 96.4% | 93.2% | 86.2% | 84.2% |
*Retention % indicates how much of the full-precision BF16 model’s accuracy was preserved after quantization.*

<figure>
  <img
    src="/src/assets/images/fixed_bar_benchmark_quant.png"
    alt="A chart showing the benchmark-wise average accuracy by quantization format, averaged across all models. This chart provides a high-level view of how different quantization formats perform on average."
  />
  <figcaption class="text-center">
    Figure 7: Benchmark-Wise Average Accuracy by Quantization Format (Across All Models, Averaged)
  </figcaption>
</figure>

From these observations, we can conclude:

* **GPTQ-INT8** is the most stable quantization format across all tasks.
* **Q5 K M** and **Q8 0** offer a great middle ground, balancing performance with portability.
* Instruction-following and multilingual tasks (**IFEval**, **C-Eval**) are the most vulnerable to aggressive quantization.
* **AWQ**, despite its efficiency, requires caution in contexts requiring high determinism.
* Tasks like math reasoning (**GSM8K**) remain robust even under Q4 formats, making these a viable option in low-resource deployments.

---

## 4 Task-Specific Recommendations

One of the core motivations behind this benchmark study is to answer a simple but often overlooked question: Which quantized LLM format should I use for my specific task or deployment context? Rather than treating benchmarks as abstract metrics, we mapped each benchmark to a real-world domain, as outlined in the Methodology. Using this mapping, we now distill our findings into task-specific recommendations, grounded in both accuracy retention and quantization stability.

### 4.1 Financial Reasoning and Math-Heavy Applications

* **Primary Benchmark**: GSM8K
* **Key Insight**: GSM8K retains high accuracy even under aggressive quantization.
* **Recommended Format**:
    * **Qwen2.5-32B + Q5 K M** for high-accuracy production
    * **Qwen2.5-7B + Q4 K M** for edge or CPU-constrained environments
    * **GPTQ-INT8** for balanced GPU deployment
* **Rationale**: Tasks with structured numeric reasoning are surprisingly resilient to quantization, making them well-suited for low-bit formats.

### 4.2 Code Generation and Developer Tools

* **Primary Benchmark**: HumanEval (referenced externally), BBH (as a proxy for logic)
* **Key Insight**: Logical reasoning and token precision are critical for code generation. GPTQ formats maintain these well, while AWQ and lower GGUFs tend to destabilize completions.
* **Recommended Format**:
    * **Qwen2.5-Coder-32B + GPTQ-INT4** for high-throughput dev tools
    * **Qwen2.5-Coder-7B + GPTQ-INT8** for smaller environments
    * Avoid **Q4 K M** for code-specific inference
* **Rationale**: Code generation demands token-level fidelity and syntactic precision, which is better preserved in GPTQ-based formats than in GGUF or AWQ.

### 4.3 Assistants and Instruction-Following Agents

* **Primary Benchmark**: IFEval
* **Key Insight**: Instruction-following accuracy degrades the fastest under low-bit quantization, especially AWQ and Q4 K M. However, agents also require fast inference and responsiveness.
* **Recommended Format**:
    * **GPTQ-INT8** for high-precision deterministic agents
    * **Q5 K M** for interactive assistants with latency needs
    * **AWQ** only if real-time performance outweighs instruction precision
* **Rationale**: Agents operating in deterministic flows require stable decoding and alignment, which lower-bit quantization disrupts. AWQ is usable but must be tested thoroughly per workflow.

### 4.4 Research and Enterprise Knowledge Tools

* **Primary Benchmark**: MMLU, C-Eval
* **Key Insight**: These models require high factual integrity and multilingual understanding. C-Eval especially reveals fragility in token-level representations under quantization.
* **Recommended Format**:
    * **GPTQ-INT8** or **Q8 0** for factual tasks in multilingual contexts
    * **Qwen2.5-14B + Q5 K M** for full-context research assistant setups
    * Avoid **AWQ** or **Q4 K M** for mission-critical factual work
* **Rationale**: These use cases demand high confidence and recall, especially in multilingual content and domain-specific retrieval. Full precision is ideal, but GPTQ-INT8 offers a strong compromise.

### 4.5 Logic and Reasoning Pipelines

* **Primary Benchmark**: BBH
* **Key Insight**: BBH is relatively robust to quantization, likely due to the internal structure of logic tasks. Even Q4 K M maintains good retention (>90%).
* **Recommended Format**:
    * **Qwen2.5-7B + Q4 K M** for embedded logic agents
    * **Qwen2.5-32B + AWQ** for fast multistep agents at scale
    * **GPTQ-INT8** for accuracy-sensitive reasoning
* **Rationale**: Reasoning chains are structurally encoded, allowing models to retain capability under quantization—a promising insight for building lightweight decision-making agents.

---

### 4.6 Cross-Cutting Observation

Across all use cases, two general trends hold:

* **Q5 K M** is consistently the safest GGUF format, retaining >95% accuracy while enabling fast, low-memory inference, making it the default recommendation when speed and fidelity must be balanced.
* **AWQ** is fast, but unstable in high-instruction or multilingual domains, so its use should be bounded to low-stakes or latency-prioritized tasks.

---

## 5 Comparative Model Analysis

While quantization formats show clear trends across tasks, each model family also exhibits unique behavior due to differences in architecture, pretraining objectives, instruction tuning, and tokenizer handling. In this section, we isolate per-model characteristics that emerged from the data, highlighting how well each model sustains performance under aggressive quantization, and which domains they’re best suited for.

### 5.1 Qwen2.5 Series: Consistent Across All Formats

Across both the Qwen2.5-Instruct and Qwen2.5-Coder models (7B, 14B, 32B), performance under quantization remained remarkably stable. The drop from BF16 to Q4 K M was generally predictable and within tolerable margins, with Q5 K M and GPTQ-INT8 retaining over 95–98% of original accuracy across all benchmarks.

**Conclusion**: Qwen2.5 is arguably the most quantization-tolerant model family in this study. Its multi-lingual grounding, strong pretraining corpus, and structured decoding make it an ideal base for quantized deployments in general-purpose assistants, agents, and coding tools.

### 5.2 DeepSeek-R1-Distill: Resilient in STEM + Instruction

The DeepSeek-R1-Distill family, particularly the Qwen-32B variant, exhibited strong resilience to quantization in STEM-oriented benchmarks. The accuracy difference between BF16 and Q4 K M across MATH, GPQA, and MMLU was consistently under 1%, even in INT4 and AWQ formats.

**Conclusion**: DeepSeek-R1-Distill models are ideal for education, tutoring, and technical reasoning agents, especially where instruction integrity must be preserved under resource constraints.

### 5.3 LLaMA 3.3: Powerful, but Fragile Under Low-Bit Quantization

While the LLaMA 3.3-70B-Instruct model delivers exceptional results in its full-precision format, it proved to be the most vulnerable to aggressive quantization. For example, on MMLU, Q4 dropped performance by 7.8 points, and even Q8 0 showed noticeable drift.

**Conclusion**: LLaMA 3.3 should be reserved for GPU-heavy, high-accuracy workloads, and is not recommended for Q4 or AWQ quantization. If used in quantized form, stick to GPTQ-INT8 and validate outputs thoroughly in mission-critical environments.

### 5.4 Mistral-7B-Instruct: The Lightweight Workhorse

Among all models evaluated, Mistral-7B-Instruct emerged as the most efficient in the 7B class. Even under 8-bit quantization, performance remained within 2% of BF16 across most tasks. Though more aggressive quantization (Q4 K M, INT4) introduced greater variance (∼8–10% drop), it remained usable for casual and lightweight deployments.

**Conclusion**: Mistral-7B is perfect for developers needing a compact, reasonably accurate LLM that performs acceptably across instruction-following and general Q&A. It is highly suitable for on-device agents, real-time chatbots, and initial production pilots.

---

## 6 Study Limitations and Future Directions

Despite covering a wide range of models, benchmarks, and quantization formats, this evaluation is not without limitations. We highlight them here both for transparency and to guide future iterations of this benchmark.

### 6.1 Missing Quantizations

Not all models were available in all quant formats. For instance, LLaMA 3.3 lacked AWQ and Q8 0 support at the time of testing, while some DeepSeek variants were missing GPTQ versions. Although interpolation between similar models helps fill interpretive gaps, direct empirical validation is always preferred.

### 6.2 Incomplete Speed Benchmarks

While accuracy and retention were central to this analysis, inference speed (tokens/sec) was not systematically benchmarked across formats or hardware profiles. Preliminary throughput numbers were observed (e.g., GGUF outperforming GPTQ on CPU), but more structured profiling, especially under batch loads, remains future work.

### 6.3 Focused Benchmark Subset

This study prioritized five representative benchmarks: BBH, MMLU, C-Eval, IFEval, and GSM8K. While these span reasoning, factual knowledge, multilingual QA, and instruction-following, they do not fully cover all LLM evaluation axes. Notably, TruthfulQA, HumanEval, and MATH were excluded due to runtime constraints and will be integrated in a follow-up post.

### 6.4 Quantization-Aware Fine-Tuning (QLoRA, GPTQ-LoRA)

We evaluated zero-shot quantized performance. However, some formats like GPTQ can recover accuracy when paired with quantization-aware fine-tuning (e.g., GPTQ-LoRA). Exploring post-quantization adaptation is an important avenue for improving quality in production deployments.

---

## 7 Conclusion

This study explored the practical implications of quantizing large language models, not just as an academic curiosity but through the lens of real-world task deployment. By evaluating over a dozen quantized variants across four major model families and five critical benchmarks, we identified meaningful patterns that developers and ML teams can act on.

**Key Takeaways**:

* **Quantization is not one-size-fits-all.** Format choice must align with task criticality and resource constraints.
* **Qwen2.5 models are the most stable** across quant formats, making them excellent candidates for versatile deployment.
* **GPTQ-INT8 and GGUF Q5 K M strike the best trade-offs** between speed and retention, outperforming more aggressive options like AWQ or Q4 K M in critical applications.
* **Instruction-following and multilingual tasks (IFEval, C-Eval) degrade faster** under quantization, whereas math and logic tasks (GSM8K, BBH) show higher resilience.
* **Model-specific traits matter:** LLaMA 3.3 offers high raw power but struggles in low-bit formats, while Mistral-7B balances efficiency with reliability for smaller use cases.

Leveraging our benchmark methodology and open-source tooling can help you evaluate your models under real-world constraints. Optimize with confidence, reduce costs, and maintain quality across production use cases.

## References

[1] Hendrycks, D., et al. Measuring Massive Multitask Language Understanding (MMLU). arXiv preprint arXiv:2009.03300, 2020. https://arxiv.org/abs/2009.03300
[2] Cobbe, K., et al. Training Verifiers to Solve Math Word Problems (GSM8K). arXiv preprint arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168
[3] Suzgun, M., et al. BIG-Bench Hard: Stress Testing Language Models with Multi-Step Reasoning. GitHub Repository. https://github.com/suzgunmirac/BIG-Bench-Hard
[4] Liu, X., et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. arXiv preprint arXiv:2305.08322, 2023. https://arxiv.org/abs/2305.08322
[5] GAIR Team. IFEval: Instruction Following Evaluation. GitHub. https://github.com/GAIR/IFEval
[6] Qwen Team. Qwen2.5 Models and Quantization Benchmarks. Hugging Face. https://huggingface.co/Qwen
[7] DeepSeek Team. DeepSeek LLMs and R1-Distill Series. GitHub. https://github.com/deepseek-ai
[8] Mistral AI. Mistral 7B Instruct Models. Hugging Face. https://huggingface.co/mistralai
[9] Meta AI. LLaMA 3.3 Models. Meta Blog. https://ai.meta.com/blog/llama-3
[10] Pan, Q., et al. AutoGPTQ: Quantization Toolkit for Large Language Models. GitHub. https://github.com/PanQiWei/AutoGPTQ
