# LLM Mechanics 最小示例

这 12 个文件分别验证教程中的一个机制。它们不下载模型或数据，也不执行训练。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

预期最后一行：

```text
12/12 examples passed
```

| 文件 | 验证内容 |
|---|---|
| `01_bpe.py` | byte-level BPE merge 与 UTF-8 round-trip |
| `02_decoder_only.py` | pre-norm decoder block 的输入输出 shape |
| `03_causal_mask.py` | 未来位置 attention 权重严格为零 |
| `04_rope.py` | 旋转保持范数与相对位置性质 |
| `05_rmsnorm.py` | 手写 RMSNorm 与 PyTorch 对齐 |
| `06_swiglu.py` | SwiGLU 门控与 ReLU² 的差异 |
| `07_prefill_decode.py` | 多 token prefill 与单 query decode |
| `08_kv_cache.py` | KV cache 追加、显存和数值一致性 |
| `09_sampling.py` | greedy、temperature 与 top-p |
| `10_repetition_eos.py` | repetition penalty 与 EOS 停止 |
| `11_teacher_forcing.py` | input / target 的一位偏移 |
| `12_perplexity.py` | perplexity 与 tokenization 口径限制 |
