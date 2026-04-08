# Subconscious Phase 0: Test Corpus + CC Directive

---

## EXTRACTION SKILL (System Prompt for 1B)

```
You are a memory extraction system. You read conversations and output structured
JSON containing atomic memories. You do not summarize or interpret. You extract
discrete facts exactly as stated or clearly implied.

RULES:
- One memory per fact. Never combine multiple facts into one memory.
- Use the speakers words as ground truth. Store numbers exactly as stated.
- Only extract what is stated or directly demonstrated.
- When a fact updates a previous fact, extract only the newer version.
- Ignore: greetings, filler, speculative discussion reaching no conclusion,
  hedging language, compliments, meta-conversation.
- If uncertain whether something is a fact or speculation, skip it.

OUTPUT: JSON array only. No preamble, no markdown fences, no explanation.

Each object:
{
  "content": "The atomic fact in one sentence",
  "memory_type": "fact | preference | state | relationship",
  "entities": [
    "entity1",
    "entity2"
  ],
  "domain": "hardware | legal | production | personal | research",
  "confidence": "high | medium"
}

MEMORY TYPES:
- fact: Stable knowledge unlikely to change.
- preference: How the user wants things done.
- state: Current status that changes over time.
- relationship: A connection between two entities.

ENTITIES: Named things (models, hardware, software, repos, protocols, concepts).
Consistent naming: ANE not Apple Neural Engine.

DOMAINS: hardware | legal | production | personal | research

DOMAIN GUIDANCE: Hardware capabilities that dont change = fact. Measurements
depending on config = state. Settled architecture decisions = state.
```

---

## USER PROMPT TEMPLATE

```
Extract memories from this conversation window. Output JSON array only.

CONVERSATION WINDOW:
---
{chunk_text}
---

JSON:
```

---

## TEST CORPUS: 10 CHUNKS

Gold set conversations: A1, B1, C1 (Nick provides manual gold sets before grading).

---

### CHUNK A1 -- Hardware: GPT-2 Fusion Results (Main 10) [GOLD SET]

```
Human: Session Results. Full Layer Fusion DONE. FFN fusion: gen_fused_ffn_mlmodelc() in compiler.py. NeuralNetworkBuilder to xcrun compile, patch ReLU to GELU mode 19. conv+GELU+conv = 1 ANE dispatch (was 2 + CPU GELU). QKV fusion: Combined c_attn weight [768 to 2304] as single conv. Output split on CPU. Result: 73 to 37 dispatches/token. 28.9 to 135.9 +/- 3.0 tok/s (4.7x speedup). Kill test: 11/11 and 15/15 tokens match PyTorch. Llama Compatibility: SiLU Easy (Mode 25 proven), No bias Easy, GQA Easy, RoPE Easy, RMSNorm Easy, SwiGLU Medium (elementwise MUL between two conv streams), Large dims Medium (16.8M >> 1.3M aned limit).

A: Correctness first: 11/11 and 15/15 on two different prompts is solid. The fusion isnt breaking output. 135.9 tok/s needs decomposition. 7.4ms per token at 37 dispatches = 200us per dispatch round-trip. Thats only 2.2x the bare dispatch floor (93us). For GPT-2 117M small dimensions (768 hidden, 3072 FFN), plausible because actual ANE compute per fused op is near-zero at these sizes. CPU overhead collapse is probably the bigger factor, not ANE dispatch savings.
```

---

### CHUNK A2 -- Hardware: 8B on ANE and Dispatch Architecture (Main 10)

```
Human: The path to 8B on ANE: the entire model compiles to 1 dispatch. Not 168 ops. Not 88 unfused. One. 32-layer transformer block, 352 espresso layers, 1 ANE dispatch, 1 program slot. The 128-slot firmware limit is irrelevant when the whole model fits in 1 slot.

A: Before saying build it, five challenges. 1) Compilation vs execution: the agent proved aned compiles 352 layers into 1 dispatch. Did it EXECUTE? 2) KV cache: fully-fused single-dispatch model takes one input and produces one output. Where does KV cache live? 3) Weight size: Llama-8B at FP16 is 16GB. Apples largest known .hwx is 184MB. 16GB is 87x larger. 4) DRAM throughput: even at 1 dispatch, 16GB FP16 through 111 GB/s = 144ms minimum = 7 tok/s ceiling. Physics doesnt change with fusion. 5) Quantization: 7 tok/s at FP16 is the floor.

H: Subconscious can work right now with 8B, right?

A: Yes, minus continuous learning. 8B at 11 tok/s is fast enough for always-on background inference, entity tracking, context compression, semantic search over vault, document summarization. It processes 660 tokens per minute continuously. A typical vault document is 500-1000 tokens. It can read, summarize, and index one document per minute while GPU runs 70B.
```

---

### CHUNK A3 -- Hardware: ANE Dispatch and Compiler Architecture (Main 10)

```
Human: So, if I understand correctly, with SIP on and no swap, we send uncompiled code with the limited 37 ops to aned for compilation. We dont include other ops because aned wont compile those. With cache swap, we compile all ops because aned isnt involved in compilation and cant limit us to the 37 ops?

A: Close, but tighter terminology. The 37 dispatches arent 37 different opcodes. Theyre 37 ops (12 QKV projections + 12 O projections + 12 fused FFN + 1 lm_head). They all use opcodes aned knows: conv/inner_product and activation mode 19. For GPT-2, aned can compile everything. The limitation is about what TYPES of operations you can express. Through aned: inner_product, conv, activation (12 modes), softmax, batchnorm, concat, reshape, elementwise (subset). Missing: 6 comparison ops (causal masking), custom PWL, INT8 dispatch path, about 45 opcodes the hardware accepts but aned has no espresso mapping for. The swap unlocks everything aned cant compile: causal attention masking, Mish/trainable PWL, INT8 weight dispatch, correct softmax masking.
```

---

### CHUNK B1 -- Production: Combined Stack Results (Main 11) [GOLD SET]

```
Human: 50.2 tok/s combined, up from 42.2 baseline. +19% end-to-end. All 4 configs 15/15 kill test PASS. The two optimizations stack cleanly: C CPU ops alone: 42.2 to 46.2 (+9.5%), slashed CPU from 2.69ms to 0.46ms (6x faster). Cross-layer fusion alone: 42.2 to 45.5 (+7.8%), ANE dropped from 16.73ms to 15.44ms. Combined: 42.2 to 50.2 (+19.0%), nearly additive. Per-token budget at combined config: ANE (16 cross-layer + 2 standalone) 15.51ms 75.5%, lm_head (8 chunks) 4.62ms 22.5%, CPU (RMSNorm + RoPE + GQA) 0.40ms 1.9%. CPU is now noise at 0.4ms. Bottleneck is firmly ANE dispatch time (15.5ms) and lm_head (4.6ms). lm_head alone is 22.5% of per-token time, those 8 chunks dispatching 128K vocab are expensive.

A: 50.2 clears the target. The 8-chunk lm_head might already be optimal. 128K vocab at dim 2048, Q8 = 256MB total weight. 8 chunks = 32MB per chunk. Thats your SRAM cliff boundary exactly. Fusing to 4 chunks means 64MB per chunk, above the cliff. Youd save dispatch overhead but each surviving chunk takes 4x longer from DRAM streaming. Net result: probably slower. Vocab pruning is the real lm_head lever. 128K to 50K shrinks weights from 256MB to 100MB. 3-4 chunks under the SRAM cliff. 4.62ms to maybe 1.8-2.2ms. But question is what Subconscious actually needs. If doing embedding: skip lm_head entirely. 15.51 + 0.40 = 15.91ms = 62.9 tok/s for embedding-mode inference.
```

---

### CHUNK B2 -- Production: Full Inventory and Architecture (Main 11)

```
Human: Honest inventory. What works: GPT-2 on ANE production-ready 229 tok/s (25d), zero GPU. Llama-1B on ANE production-ready 50.2 tok/s (25d), zero GPU, C+fusion. Llama-1B agent server working on :8423, 27.3 tok/s (old config, needs rebench). Llama-8B on ANE prototype only 13.3 tok/s (V-to-O shortcut, no real attention). ane-dispatch shipped GitHub MIT. ane-compiler shipped GitHub MIT. Direct kext port proven, survives aned death. MIL IR fusion proven, 2-input arbitrary op composition. What doesnt connect: the 70B spec decode server on :8899 uses 1B Q8 on CPU/AMX as drafter at 10.3ms/tok. The ANE 1B is at 50.2 tok/s = 20ms/tok. CPU drafter is still 2x faster. ANE 1B was already killed as drafter (same model = same predictions, 0-1% unique acceptance). ANE 1B path is a parallel independent workload: agent, classification, embedding, zero GPU contention.

A: This is the clearest picture youve put together. The 8B math needs a reality check. 8B Q8 is 8GB weights. ANE DMA at 111 GB/s = 72ms minimum weight streaming per token = 14 tok/s theoretical ceiling. With paged dispatch (2 loads of 84 programs), adding a second round-trip through aned per token. Realistic estimate: 8-12 tok/s. The 8B prize isnt throughput. Its capability. An 8B model doing continuous context compression, entity extraction, document classification is qualitatively different for Subconscious.
```

---

### CHUNK B3 -- Production: Server Architecture (Main 5/6)

```
Human: Give CC a full rundown of the production framework with each responsibility.

A: PRODUCTION ARCHITECTURE. COMPONENT 1: 70B Q3 Verifier (GPU). Llama 3.3 70B Q3 on MLX, Port :8899. Role: generates verified tokens, accepts/rejects draft candidates from speculative decode. COMPONENT 2: 1B CPU Drafter (AMX). Llama 3.2-1B on CPU via AMX wide-load path. Runs inside the spec decode server process. Role: generates draft tokens at 10.3ms/tok when N-gram hit rate drops below adaptive threshold (currently 15%). 2.7% GPU contention (proven independent silicon). COMPONENT 3: 3B ANE Daemon (Neural Engine). Llama 3.2-3B on ANE via CoreML. Role: autonomous research agent, background document analysis, vault monitoring, semantic retrieval. NOT a drafter, independent workload. 24.6 tok/s, 1.0% GPU contention at timed dispatch. COMPONENT 4: N-gram Table (software). Builds from 70B output tokens during generation. Adaptive threshold: suppresses N-gram drafting when rolling acceptance drops below 15%. When hot (>15%): drafts at near-zero cost, drives 21+ tok/s peaks. When cold: yields to 1B CPU drafter. ORCHESTRATION: N-gram drafts first. If acceptance < 15%: suppress, CPU 1B takes all draft slots. 70B verifies all candidates. ANE daemon runs independently.
```

---

### CHUNK C1 -- Architecture: Subconscious Memory Manager (Main 11) [GOLD SET]

```
Human: Lets take a step back, what is the true role of an always online sub agent? I would think its a subconscious memory manager, pulling the right memories for the conversation, organizing memories, trimming/removing stale memories, becoming smarter so it knows which memories are relevant. It feels to the user that the 70B model is growing, adapting, context aware with a real memory.

A: Thats the reframe that was missing. Everything we specced (embedding models, classification, entity graphs) was bottom-up: we have this compute lane, what do we put on it? Youre now asking top-down: what does the user experience? The user experience: they talk to the 70B. The 70B remembers. Not because someone said remember this, but because it noticed. It surfaces things you forgot you told it. It never surfaces irrelevant noise. The user never sees Subconscious. They never interact with it. They just notice the 70B is uncannily good at context. Thats a fundamentally different product than an agent daemon. The memory lifecycle has four phases: Ingestion (extract atomic facts from every exchange), Storage (structured memory store with content, timestamp, source, entity links, relevance decay, access count, embedding), Retrieval (before every 70B generation, answer what memories matter given what the user just said), Maintenance (consolidation, contradiction resolution, decay, strengthening). The getting smarter part: memory store grows passively, relevance scores adapt through usage statistics, extraction skill can improve via 70B feedback during idle time.
```

---

### CHUNK C2 -- Architecture: Compute Mapping and Hot/Cold Loops (Main 11)

```
Human: From CC: The hot loop budget is tighter than 19ms. The 70B first-token latency is already 95ms (TTFT on Q3). Retrieval is AMX embedding + cosine similarity over structured store, genuinely sub-millisecond for a few thousand memories. The budget isnt the problem. The prompt assembly is. How many tokens of memory context can you inject before 70B generation quality degrades from context pollution? Start with 5 memories max (maybe 500 tokens) and tune from there.

A: CCs build order is right. One correction: Phase 2 does not replace vault_read. The vault is ground truth. CLAUDE.md is canonical session memory. Subconscious memory is a derived store that supplements it. If Subconscious retrieves a stale memory that contradicts the vault, the vault wins. Architecture: read vault (ground truth) + retrieve memories (supplemental context), inject both into 70B prompt. Addition 1: Phase 0 needs a designed extraction prompt, not just run through 1B. The Lotto Pattern: Claude writes the skill, local model executes. The prompt is the product, not the model. Addition 2: Grading rubric needs to be defined before the test. Three metrics: precision (what fraction of extractions are worth storing), recall (what fraction of gold set was found), poison rate (what fraction is factually wrong). Precision > 90%, recall > 70%, poison < 5%. Poison matters most because a wrong memory injected into 70B produces confidently incorrect responses.
```

---

### CHUNK C3 -- Architecture: Multi-Model Subconscious Design (Main 11)

```
Human: Which model is best at providing 70B context?

A: Challenge the assumption: does providing context require generation? The 70B doesnt need pre-digested summaries. Its Tier 3. Give it raw relevant chunks and it synthesizes. The question becomes: whats the best way to select and retrieve the right chunks. Task 1 Retrieval: dedicated embedding model crushes any decoder. nomic-embed-text (137M params), GTE-small (33M), all-MiniLM-L6 (22M). Trained with contrastive objectives specifically for retrieval. Task 2 Classification: 1B well-suited. Lotto Pattern proven. Binary and multi-class doesnt need deep reasoning. Task 3 Context compression: 1B generation quality for summarization is marginal. Options: use 70B itself during idle time (perfect quality, zero inference cost), use 8B on ANE when paged dispatch lands, or skip compression entirely and rely on retrieval. Architecture reframes to multi-model: AMX runs embedding model (200M, 2ms) for retrieval + entity matching. ANE runs Llama-1B (50 tok/s, 16ms embed mode) for classification + extraction. GPU runs 70B for generation + idle-time compression. CPU does orchestration + N-gram. Total Subconscious latency: 18ms under 19ms budget.
```

---

### CHUNK C4 -- Architecture: AMX Role in Production (Main 11)

```
Human: What are we going to run on the AMX?

A: Right now, nothing, and that should bother you because its your third zero-contention compute lane sitting idle. AMX is fast when working set fits in L2 (16MB per S-core cluster). Anything larger falls to DRAM at 307 GB/s shared with GPU. So the question isnt what model runs on AMX but what 16MB-or-less workload is useful. Candidate 3 is the natural fit: Subconscious entity operations. Entity embeddings at dim=256 for 1000 entities = 1MB. Cosine similarity matrix for retrieval = trivially small. Entire entity retrieval and ranking fits in L2. AMX at 900 GB/s doing entity matching while ANE runs 1B model and GPU runs 70B = all three accelerators active on every token. Retrieval step at 900 GB/s is sub-microsecond for 1MB entity store. The three-lane picture: GPU (70B decode) + ANE (1B Subconscious: draft generation, context compression, distillation) + AMX (entity retrieval, graph operations, potentially LoRA application). CPU orchestrates and runs N-gram as fallback.
```

---

## GRADING PROTOCOL

### Metrics (per gold-set conversation, then averaged):

- **Precision** = correct extractions / total extractions by 1B. Target: > 90%
- **Recall** = gold set items found / total gold set items. Target: > 70%
- **Poison** = factually wrong extractions / total extractions. Target: < 5%
- **Malformed JSON** rate. Target: < 10%

### Go/No-Go:

- PASS: Precision > 90% AND Poison < 5% AND Malformed JSON < 10%
- ITERATE: Precision 75-90% OR Poison 5-10% (prompt needs work)
- FAIL: Precision < 75% OR Poison > 10% (1B insufficient)

### Grading template per gold-set chunk:

```
Chunk: [A1/B1/C1]
Domain: [hardware/production/architecture]
Gold set size: [N memories]
1B extractions: [M memories]
Correct: [X]  Wrong (poison): [Y]  Irrelevant: [Z]
Gold items found: [W]
Precision: X/M = ____%
Recall: W/N = ____%
Poison: Y/M = ____%
Malformed JSON: [yes/no]
Notes: [specific failure patterns]
```

---

## CC DIRECTIVE

```
Read vault/subconscious/phase0_package.md. Execute Phase 0.

Step 1: Chunk preparation.
The 10 chunks (A1-A3, B1-B3, C1-C4) are already in the file.
Save each chunk as a separate file:
  vault/subconscious/chunks/chunk_A1.txt
  vault/subconscious/chunks/chunk_A2.txt
  ... etc.
Strip the chunk headers. Keep only the Human:/Assistant: content.

Step 2: Gold sets.
WAIT for Nick to provide gold sets for A1, B1, C1 as JSON files:
  vault/subconscious/gold_sets/gold_A1.json
  vault/subconscious/gold_sets/gold_B1.json
  vault/subconscious/gold_sets/gold_C1.json
DO NOT PROCEED past Step 2 until gold sets exist.

Step 3: Run extraction.
For each chunk, send to Llama-1B on ANE (port :8423, 25d+C config):
  System prompt = EXTRACTION SKILL from this file
  User prompt = USER PROMPT TEMPLATE with chunk text substituted
Save raw JSON output per chunk:
  vault/subconscious/extractions/extract_A1.json
  vault/subconscious/extractions/extract_A2.json
  ... etc.
If 1B produces malformed JSON, log the failure and save raw output.

Step 4: Grade gold-set chunks (A1, B1, C1).
Compare extractions against gold sets using the grading template.
For remaining 7 chunks: qualitative scan for hallucinated facts,
missed critical info, entity naming consistency, memory type accuracy.

Step 5: Report.
Single results table with all grading metrics.
Go/No-Go determination.
List of specific failure patterns if any.
Do not interpret or recommend. Report the numbers.
```

---

## FOR NICK: Before handing to CC

1. Save this file to vault/subconscious/phase0_package.md
2. Make sure ane_server_v2 is running on :8423 with 25d+C config
   (if still on old config, wire the combined stack first)
3. Write gold sets for chunks A1, B1, C1:
   - Read each chunk
   - List every memory you would want stored as JSON objects
   - matching the extraction schema
   - 15-20 memories per chunk is the right target
   - Do this BEFORE running 1B so you dont anchor to its output
4. Save gold sets to vault/subconscious/gold_sets/
5. Tell CC: Read vault/subconscious/phase0_package.md. Execute Phase 0.
