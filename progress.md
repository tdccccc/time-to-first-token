---
name: Progress tracker
about: Track your way through the roadmap. Open this in your own fork.
title: 'Roadmap progress: [your name or handle]'
labels: progress
---

Tick sessions as you finish them. GitHub saves checkbox state to your account, so this works across every device you sign in from.

Started on:

### Week 1: The mental model
- [ ] Mon. Making Deep Learning Go Brrrr, first half
- [ ] Tue. Making Deep Learning Go Brrrr, second half
- [ ] Wed. CS336 Lecture 5, GPUs, first 30 min
- [ ] Thu. CS336 Lecture 5, second 30 min
- [ ] Fri. Hand-derive arithmetic intensity for your model

### Week 2: vLLM
- [ ] Mon. PagedAttention announcement and paper
- [ ] Tue. Serve a 7-8B model behind vLLM
- [ ] Wed. vLLM V1 architecture
- [ ] Thu. Inside vLLM, scheduler and block manager
- [ ] Fri. Follow the block table code path

### Week 3: Measurement
- [ ] Mon. vLLM metrics design doc
- [ ] Tue. Stand up Prometheus and Grafana
- [ ] Wed. Import the dashboard, confirm live panels
- [ ] Thu. Map every metric to a panel
- [ ] Fri. Modal benchmarking methodology

### Week 4: SGLang and batching
- [ ] Mon. RadixAttention
- [ ] Tue. Deploy SGLang with the same model
- [ ] Wed. SGLang talk and paper
- [ ] Thu. Continuous batching and Orca
- [ ] Fri. Sarathi-Serve, chunked prefill

### Week 5: Load testing
- [ ] Mon. GuideLLM intro and install
- [ ] Tue. First GuideLLM sweep
- [ ] Wed. vllm bench serve cross-check
- [ ] Thu. genai-perf concurrency sweep
- [ ] Fri. Push past 1000 concurrent

### Week 6: Quantization
- [ ] Mon. MIT 6.5940 Lecture 5, first half
- [ ] Tue. Lecture 5 and 6, AWQ, GPTQ, SmoothQuant
- [ ] Wed. Quantization survey, map methods to engines
- [ ] Thu. Serve AWQ or GPTQ and FP8 variants
- [ ] Fri. Quality proxy across FP16, FP8, INT4

### Week 7: Speculative decoding and KV eviction
- [ ] Mon. Speculative decoding docs and benchmark post
- [ ] Tue. GPU MODE Lecture 22, first half
- [ ] Wed. Benchmark speculative decoding at low and high QPS
- [ ] Thu. StreamingLLM
- [ ] Fri. Long-context eviction, measure memory and TTFT

### Week 8: Disaggregation and Kubernetes
- [ ] Mon. DistServe talk and paper
- [ ] Tue. Splitwise and vLLM disaggregated prefilling
- [ ] Wed. Mooncake and SGLang PD disaggregation
- [ ] Thu. Deploy with the production stack
- [ ] Fri. Autoscale on queue depth, not CPU

### Week 8 (continued): Large-scale cluster PD optimization
- [ ] Mon. Mooncake deep dive: Conductor, KV pool, early rejection
- [ ] Tue. KV transfer math: size, bandwidth, share of TTFT
- [ ] Wed. Cross-node KV pooling and reuse
- [ ] Thu. Multi-node disaggregated deployment, disaggregated vs colocated
- [ ] Fri. Pool sizing, autoscaling signals, fault tolerance

### Week 9: Unit economics and the router
- [ ] Mon. Inference economics from first principles
- [ ] Tue. Re-derive cost per million tokens on your own GPU
- [ ] Wed. Character.AI inference posts
- [ ] Thu. Build the router
- [ ] Fri. Token budgeting middleware

### Week 10: Publish
- [ ] Mon. Draft the benchmark writeup
- [ ] Tue. Add labeled variants, publish
- [ ] Wed. Set up the weekly reading habit
- [ ] Thu. First habit session
- [ ] Fri. Optional edge sampler

### Week 11: TensorRT-LLM (optional)
- [ ] Mon. Architecture: PyExecutor loop, IFB, paged KV, scheduler budgets
- [ ] Tue. Deploy the same model behind trtllm-serve
- [ ] Wed. FP8 quantization, measure with the M0 harness
- [ ] Thu. EAGLE-3 speculative decoding, KV reuse check, disagg config
- [ ] Fri. Three-engine comparison report and engine decision guide

### Pre-publish checklist
- [ ] Swept the request rate
- [ ] Reported exact input and output token lengths
- [ ] Reported p50, p95, p99
- [ ] Kept TTFT and inter-token latency separate
- [ ] Did not measure at the preemption point
- [ ] Used realistic, not uniform, length distributions
- [ ] Accounted for tokenizer differences
- [ ] Pinned versions and published exact commands

---

If you finish, link your benchmark writeup here. Open a PR to add it to the README.

---

本 fork 新增模块：M0 基础与度量、M2 算子优化、M3 国产芯片适配、M4 综合项目，见 [docs/MODULES.md](docs/MODULES.md)；它们的进度用各自模块 README 里的清单跟踪。
