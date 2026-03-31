# IsoQuant

IsoQuant is a blockwise rotation quantization prototype based on quaternion algebra and the isoclinic decomposition of `SO(4)`. It replaces RotorQuant's 3D Clifford blocks with hardware-aligned 4D quaternion blocks and currently focuses on the stage-1 quantize-dequantize path for LLM KV-cache compression.

## Current Status

This repository is an early research prototype.

- Implemented: `IsoQuant-Full`, `IsoQuant-Fast`, and a lightweight `IsoQuant-2D` special case; quaternion utilities; fused CUDA kernels; and benchmark scripts.
- Measured so far: reconstruction MSE and fused CUDA kernel latency on synthetic normalized vectors.
- Not yet validated: end-to-end KV-cache quality on real model activations, attention-logit fidelity, retrieval accuracy, or perplexity.

Please read the current results with that scope in mind: the evidence here supports a stage-1 MSE and systems claim, not a complete KV-cache compression claim yet.

## Main Idea

TurboQuant uses a dense random orthogonal rotation before scalar quantization, which costs `O(d^2)` parameters and arithmetic. RotorQuant reduces this cost with 3D Clifford-algebra blocks, but 3D chunking is awkward for common LLM head dimensions such as `64`, `128`, and `256`.

IsoQuant instead partitions the feature into 4D blocks and applies an `SO(4)` rotation parameterized by one or two unit quaternions:

- `IsoQuant-Full`: `v -> q_L v conjugate(q_R)`
- `IsoQuant-Fast`: `v -> q_L v`
- `IsoQuant-2D`: a lightweight planar special case on 2D coordinate pairs

The main method is still the 4D construction: `IsoQuant-Full` provides the strongest local mixing, while `IsoQuant-Fast` lowers cost by keeping only one isoclinic factor. `IsoQuant-2D` is included as an auxiliary low-cost operating point rather than a separate method family.

## Variants

| Variant | Block type | Main transform | Role |
| --- | --- | --- | --- |
| `IsoQuant-Full` | 4D quaternion block | `v -> q_L v conjugate(q_R)` | Primary and most expressive variant |
| `IsoQuant-Fast` | 4D quaternion block | `v -> q_L v` | Lower-cost 4D variant |
| `IsoQuant-2D` | 2D planar block | `u -> R(theta) u` | Lightweight special case |

## Current Results

All fused CUDA measurements below were collected on a single NVIDIA RTX 4090 GPU. These are stage-1 synthetic benchmarks only.

| Method | Average speedup vs RotorQuant fused CUDA | Peak speedup | Quality scope |
| --- | ---: | ---: | --- |
| `IsoQuant-Full` | `4.49x` | `5.92x` | Reconstruction MSE only |
| `IsoQuant-Fast` | `4.66x` | `6.31x` | Reconstruction MSE only |
| `IsoQuant-2D` | `4.66x` | `6.39x` | Reconstruction MSE only |

These averages come from `18` fused CUDA benchmark settings spanning:

- `d in {128, 256, 512}`
- `bits in {2, 3, 4}`
- `dtype in {fp16, fp32}`

Across these settings, all three IsoQuant variants maintain reconstruction MSE that is essentially identical to RotorQuant at the reported precision.

At `d=128`, the forward rotation complexity is:

| Method | Block structure | Parameters | FMAs |
| --- | --- | ---: | ---: |
| TurboQuant | dense `128 x 128` | `16,384` | `16,384` |
| RotorQuant | `43 x 3D` blocks | `172` | `~2,408` |
| `IsoQuant-2D` | `64 x 2D` blocks | `128` | `~256` |
| IsoQuant-Full | `32 x 4D` blocks | `256` | `1,024` |
| IsoQuant-Fast | `32 x 4D` blocks | `128` | `512` |

The 2D case is the cheapest operating point, but the 4D formulation remains the main method because it offers stronger local mixing and cleaner alignment with the paper's `SO(4)` motivation.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `isoquant/isoquant.py` | Main `IsoQuantMSE` implementation |
| `isoquant/planar2.py` | Lightweight 2D special-case quantizer |
| `isoquant/quaternion.py` | Quaternion math and block packing |
| `isoquant/csrc/` | CUDA kernels for fused stage-1 quantization |
| `isoquant/benchmark_cuda.py` | CUDA benchmarks against RotorQuant |
| `isoquant/benchmark_metal.swift` | Swift/Metal benchmark prototype |
| `isoquant/validate_isoquant.py` | Minimal validation script |
| `latex/isoquant.tex` | Paper draft source |
| `rotorquant/` | Baseline code used for comparison |

## Quick Start

Run the lightweight validation:

```bash
PYTHONPATH=. python -m isoquant.validate_isoquant
```

Run CUDA benchmarks:

```bash
PYTHONPATH=. python -m isoquant.benchmark_cuda
```

The CUDA benchmark script JIT-compiles the extension on first use and expects a CUDA-capable PyTorch environment. It reports fused CUDA results for RotorQuant, `IsoQuant-Full`, `IsoQuant-Fast`, and `IsoQuant-2D`.

Example sweep:

```bash
LOG=isoquant_cuda_sweep.log
for t in fp16 fp32; do
  for d in 128 256 512; do
    for b in 2 3 4; do
      echo "===== dtype=$t bits=$b dim=$d =====" | tee -a "$LOG"
      PYTHONPATH=. python -m isoquant.benchmark_cuda --batch-size 8192 --dim $d --bits $b --dtype $t 2>&1 | tee -a "$LOG"
      echo "" | tee -a "$LOG"
    done
  done
done
```

## Citation

```bibtex
@misc{ji2026isoquanthardwarealignedso4isoclinic,
      title={IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression}, 
      author={Zhongping Ji},
      year={2026},
      eprint={2603.28430},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.28430}, 
}
```

## Next Steps

- Add real KV-cache extraction and evaluation on deployed LLMs
- Integrate a stage-2 residual correction path such as QJL
- Measure attention-logit fidelity, retrieval metrics, and perplexity
- Extend the evaluation from stage-1 synthetic benchmarks to full end-to-end conference-quality experiments
