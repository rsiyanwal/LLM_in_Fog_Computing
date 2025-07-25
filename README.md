# A brief review of the use of LLM in Fog Computing
For the time being, let's assume that LLM models (referred to as just "models" in the rest of the text) are black boxes. We will focus on the input these models receive and how they generate output:
1. **Input (what exactly does a model receive?):** At present, fine-tuned models such as ChatGPT can "remember" the conversation/context. We are aware that a neural network, however complicated it becomes, does not have an in-built memory, so how is this "memory" implemented? Every time we send a message, the model is unaware of the past conversation unless explicitly provided again. The trick is that the system keeps a record of the conversation in the background and includes some part of it (last `n` tokens) with each new message. Furthermore, the input also contains a _system prompt_ (instructions about the model's role).
2. **Output:** Given this input, the model's job is to predict the best next token, which is appended to the conversation and fed back to predict the best next token again. This loop continues till the model emits an end-of-sequence token or reaches the length limit.

## Language models available today

## A breif introduction to top AI platforms
We will investigate the architectures of ChatGPT, Perplexity AI, and Google Gemini. They all share `Transformer` foundation, but each platform differs from one another to align with their goals: ChatGPT for a general-purpose chat utility, Perplexity AI for real-time search with references and citations, and Gemini for multimodal tasks. Generally, these companies keep their architecture closed. However, many things mentioned below are known as fact, while a few are based on rumors (which will be mentioned clearly). 
### ChatGPT (OpenAI)
- Built on proprietary GPT series (GPT 3.5, GPT 4, and GPT 4o), with native support for text, image, and audio (making it multimodal too).
- Evolved from a `decoder`-only Transformer.
- Priority: `agentic` behaviors and broad conversation capabilities.
- ChatGPT is a fine-tuned version of these models using techniques like RLHF
According to leaks, GPT 4 has ~1.7 trillion parameters, and the architecture is rumored to use a [Mixture-of-Experts (MoE)](https://huggingface.co/blog/moe) design. The 1.7 trillion parameters are composed of 8 or 16 experts (each with ~100-200B parameters), of which only a subset activate per query to manage the cost.
### Perplexity AI
Perplexity AI, a search assistant, orchestrates a mix of third-party and its own LLMs and even lets users choose the preferred model. Perplexity can use OpenAI’s `GPT 4`, Anthropic’s Claude (`v3.5 Sonnet` is mentioned), a model called `Grok-2` (xAI, formerly Twitter), and a `Llama 3` (Meta) based on the best-suited model for the task. Perplexity [runs](https://aws.amazon.com/solutions/case-studies/perplexity-bedrock-case-study/) on AWS's Bedrock and Sagemaker infrastructure. They also fine-tuned open-source models to create a series of `Sonar` [models](https://sonar.perplexity.ai/) and PPLX [models](https://www.perplexity.ai/hub/blog/introducing-pplx-online-llms) based on Llama 3.3, Mistral 7B (and a mixture of 7B experts), and other open-source models. Their emphasis is on real-time, cited answers. They require a highly optimized [Retrieval Augmented Generation (RAG) system](https://medium.com/research-highlights-by-winston-wang/how-rag-technology-powers-ai-driven-search-engines-a-deep-dive-into-tech-behind-perplexity-ai-252f8fe4f197) with robust web crawling and low-latency inference.
### Google Gemini
[Gemini](https://blog.google/technology/ai/google-gemini-ai/#performance) is one of the series of [foundational models](https://cloud.google.com/vertex-ai/generative-ai/docs/models) from Google. Gemini is a native multimodal Transformer model that was pre-trained from the start on different modalities such as text, images, code, and audio (In case of ChatGPT, GPT 4 and GPT 4o are multimodal). 

### RAG
- All platforms (especially Perplexity) use RAG to response with external, up-to-date data.
- They use token embeddings using models like gemini-embedding-001 or OpenAI embeddings.
- Such embeddings are stored in vector databases, such as for similarity search.
### Context and session management
- ChatGPT uses history or summarization with max token size of 128K tokens.
- Perplexity uses attention-based memory with sessions.
- Gemini supports upto 2M tokens window, uses context-caching.

| Feature                   | ChatGPT                       | Perplexity AI                     | Google Gemini                    |
| ------------------------- | ----------------------------- | --------------------------------- | -------------------------------- |
| **Primary Focus**         | Conversational AI, Agentics   | Real-time, factual search         | Unified multimodal reasoning     |
| **RAG System**            | Optional (e.g., via plugins)  | Core architecture                 | Embedded in future pipelines     |
| **Citation/Transparency** | Limited                       | Inline citations by default       | Planned, not primary feature     |
| **Multimodality**         | GPT-4o supports it            | Not native                        | Fully native multimodal          |
| **Context Size**          | Up to 128K tokens             | Session-based + memory            | 2M tokens (Gemini 1.5 Pro)       |
| **Reasoning & Planning**  | Task agents (e.g., File Tool) | Multi-step plans in Deep Research | Deep Think, async agents         |
| **Hardware**              | Azure + NVIDIA GPUs           | AWS + Cerebras + Vespa            | GCP + TPUs                       |

## LLM in Fog Computing
LLM models due to their shear size thrive in cloud data centers with abundant GPU memory and computing capabilities. Running such models on resource-constraint devices require significant optimizations:
### Smaller, efficient Transformer variants 
Fog devices have strict memory constraints (often 4 - 8 GB RAM) and compute. Standard Transformers are too heavy for such devices. Some of the optimizations are as follow:
1. **Self-attention:** [Linformer](https://arxiv.org/abs/2006.04768) approximates the [attention-matrix](https://medium.com/@gginis/the-attention-mechanism-in-deep-learning-an-example-fb6b27c30cff) by a [low-rank](https://www.ibm.com/think/topics/lora#:~:text=This%20method%20focuses%20on%20the,computing%20power%20and%20training%20time) product, cutting complexity to a linear time with minimal accuracy loss.
2. **Replacing self-attention layer:** [FNeT](https://www.geeksforgeeks.org/deep-learning/fnet-a-transformer-without-attention-layer/), a Transformer without an Attention layer yeilds model that train ~70-80% faster than [BERT](https://arxiv.org/abs/1810.04805) with only slight accuracy degradation.
3. **Parameter sharing:** Another design approach is to reduce total parameter count while retaining depth of the model. [ALBERT](https://arxiv.org/pdf/1909.11942) demonstrates two such techniques: 1) Factorized embeddings and 2) Cross-layer parameter sharing.
4. **Sparse Mixture-of-Experts (MoE) Models:** The difference between a sparse and a dense MoE lies in their activation and computation during the inference. A sparse MoE activates only a small subset of it's sub-network (experts) for each input token whereas dense MoE considers all experts. Generally, models are trained across all experts (dense MoE) but utilizing sparse activation for the inference - [Dense training, Sparse Inference](https://arxiv.org/abs/2404.05567). Examples: Alibaba’s [Qwen-1.5 MoE](https://qwenlm.github.io/blog/qwen-moe/#:~:text=Introduction,resource%20utilization%20without%20compromising%20performance.) (14.3B total parameters, top-2 experts active) uses only 2.7B parameters per inference and outperforms dense 7B models. Another example is Contextual AI's [OLMoE](https://openreview.net/forum?id=xXTkbTBmqq) where only 1B parameters are active per token claims that the model can [easly run on edge devices](https://contextual.ai/olmoe-mixture-of-experts/#:~:text=By%20Niklas%20Muennighoff,devices%2C%20vehicles%2C%20IoT). 
5. **Weight offloading:** Allows storing parts of LLM in a cheaper, slower memory (CPU RAM or SSD) and loading them to GPU on demand - [Source](https://dl.acm.org/doi/10.1145/3719330.3721230). 
### Device-specific architecture choices
1. **SBCs:** For devices like Raspberry Pi, the model must be extremely compact. Projects like [llama.cpp]() [demonstrate](https://github.com/ggml-org/llama.cpp/issues/58) running 7B parameter Transformer with a 4-bit integer weight on a Pi 4 (4 GB). Realistically, 2-3B parameter models can be [used](https://itsfoss.com/llms-for-raspberry-pi/#:~:text=As%20you%20can%20see%20in,Raspberry%20Pi%205%20was%20impressive) on a pi-class hardware.
2.  **GPU-enabled edge devices:** Devices such as Jetson Orin can host large models as compared to Pis. For example, Nvidia demonstrated models up to 7B parameters [running](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md) on Jetson Orin using TensorRT optimization.
3. **Memory bandwidth reduction:**  Multi-Query Attention (MQA) reduces memory space and bandwidth needed for inference computation by [sharing key and value head across all query heads](https://fireworks.ai/blog/multi-query-attention-is-all-you-need).
4. **Grouped-Query Attention (GQA) as a MQA Extension:** GQA is an interpolation between Multi-headed Attention (MHA) and MQA. Mistral 7B uses GQA for [faster inference](https://www.datacamp.com/tutorial/mistral-7b-tutorial).
### Optimization techniques
#### Model compression methods
1. **Parameter-weight Quantization:** By using fewer bits to represent weights, model size drops drastically. Post-training Quantization (PTQ) methods such as GPTQ can quantize weight to 3-4 bit integer while calibrating to minimize the errors induced in the layer's output. The [result](https://arxiv.org/abs/2504.02118) is 3-4.5x faster inference on GPUs with minimal accuracy loss.
2. **Activation-weight quantization:** It observes that not all weights are critical. By protecting just 1% of sensitive weights [AWQ (Activation-aware Weight Quantization)](https://arxiv.org/abs/2306.00978) can quantize the rest aggressively.
3. **Weight pruning:** Methods such as [SparseGPT](https://arxiv.org/pdf/2301.00774) removes less important connections and updates the remaining weights.
4. **Structure pruning:** This means removing entire neuron, attention head or even layer to shrink the model. For instance, [LLM Pruner](https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html) identifies "non-critical coupled structures" and prunes them out. LLM Pruner pruned LLaMA-7B by [~20%](https://blogs.novita.ai/unveiling-llm-pruner-techniques-doubling-inference-speed/#:~:text=Referencing%20the%20table%20below%2C%20pruning,18%25%20increase%20in%20inference%20speed.) with virtually no performance loss after a brief fine-tuning. Other examples are: [Tailored-LLaMA](https://arxiv.org/html/2410.19185v2#:~:text=Firstly%2C%20how%20can%20we%20optimize,prompts%20consistently%20yields%20higher%20accuracy.) and [SlimGPT](https://neurips.cc/virtual/2024/poster/95477#:~:text=However%2C%20SlimGPT%20with%20finetuning%20still%20leads%20on%20most%20of%20the%20tasks.&text=PPL%20&%20Commonsense%20Reasoning%20&%20MMLU%20evaluations,solely%20on%20lightweight%20LoRA%20finetuning.).
5. Knowledge distillation
6. LoRA
#### Inference optimization techniques
#### Trade-offs and performance analysis
#### Fog-based LLM applications

