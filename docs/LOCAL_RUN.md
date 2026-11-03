# Running locally on a laptop RTX 3050 (4 or 6 GB VRAM)

Check VRAM first: `nvidia-smi`.  4 GB -> <=3B 4-bit models; 6 GB -> up to ~7-8B, tight.
GPT-OSS-120B (MPBench's backbone) will NOT run locally, so results are not comparable to the paper.

```bash
git clone https://github.com/sykeriin/TaintMem && cd TaintMem && git checkout taintmem-core
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                                          # numpy only
python -m unittest discover -s tests                      # expect: OK (skipped=1)

# install Ollama (https://ollama.com), then:
ollama pull qwen2.5:3b
python -m evals.preflight --backbone ollama:qwen2.5:3b    # MUST say "format OK: 2/2" before anything else
python -m evals.grid --backbone ollama:qwen2.5:3b --n 20 --out results/local.jsonl
```
* If preflight reports format failures, the grid would mostly measure format errors: try a larger/instruct model.
* Reruns are free: responses are cached in `results/cache_*.jsonl`.
* Expect 15-40 tokens/s; keep `--n` small first. Send me the printed table or `results/local.jsonl`.
* Real-data checks that need no model: `python -m evals.retrieval_recall LME.json` (download
  `longmemeval_oracle.json` from the xiaowu0162/longmemeval-cleaned dataset on Hugging Face) and
  `python -m evals.memorygraft REPO_DIR` (clone Jacobhhy/agent-memory-poisoning with GIT_LFS_SKIP_SMUDGE=1).
