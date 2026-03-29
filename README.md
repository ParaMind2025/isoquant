# IsoQuant

IsoQuant is a 4D block rotation quantization prototype based on the isoclinic decomposition of `SO(4)`. It replaces RotorQuant's 3D Clifford blocks with hardware-aligned 4D quaternion blocks and currently focuses on the stage-1 quantize-dequantize path.

## Current Status

This repository is an early research prototype.

- Implemented: `IsoQuant-Full` and `IsoQuant-Fast` stage-1 quantizers, quaternion utilities, fused CUDA kernels, and benchmark scripts.
- Measured so far: reconstruction MSE and fused CUDA kernel latency on synthetic normalized vectors.
- Not yet validated: end-to-end KV-cache quality on real model activations, attention-logit fidelity, retrieval accuracy, or perplexity.

Please read the current results with that scope in mind: the evidence here supports a stage-1 MSE and systems claim, not a complete KV-cache compression claim yet.

## Main Idea

TurboQuant uses a dense random orthogonal rotation before scalar quantization, which costs `O(d^2)` parameters and arithmetic. RotorQuant reduces this cost with 3D Clifford-algebra blocks, but 3D chunking is awkward for common LLM head dimensions such as `64`, `128`, and `256`.

IsoQuant instead partitions the feature into 4D blocks and applies an `SO(4)` rotation parameterized by one or two unit quaternions:

- `IsoQuant-Full`: `v -> q_L v conjugate(q_R)`
- `IsoQuant-Fast`: `v -> q_L v`

This gives a hardware-friendly 4-wide structure that maps naturally to fused kernels and SIMD-style execution.

## Current Results

All fused CUDA measurements below were collected on a single NVIDIA RTX 4090 GPU. These are stage-1 synthetic benchmarks only.

| Method | Average speedup vs RotorQuant fused CUDA | Peak speedup | Quality scope |
| --- | ---: | ---: | --- |
| `IsoQuant-Full` | `4.83x` | `6.46x` | Reconstruction MSE only |
| `IsoQuant-Fast` | `5.08x` | `6.46x` | Reconstruction MSE only |

At `d=128`, the forward rotation complexity is:

| Method | Block structure | Parameters | FMAs |
| --- | --- | ---: | ---: |
| TurboQuant | dense `128 x 128` | `16,384` | `16,384` |
| RotorQuant | `43 x 3D` blocks | `172` | `~2,408` |
| IsoQuant-Full | `32 x 4D` blocks | `256` | `1,024` |
| IsoQuant-Fast | `32 x 4D` blocks | `128` | `512` |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `isoquant/isoquant.py` | Main `IsoQuantMSE` implementation |
| `isoquant/quaternion.py` | Quaternion math and block packing |
| `isoquant/csrc/` | CUDA kernels for fused stage-1 quantization |
| `isoquant/benchmark_cuda.py` | CUDA benchmarks against RotorQuant |
| `isoquant/validate_isoquant.py` | Minimal validation script |
| `paper/isoquant.pdf` | Paper draft |
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

The CUDA benchmark script JIT-compiles the extension on first use and expects a CUDA-capable PyTorch environment.

## Citation

```bibtex
@article{jzp2026isoquant,
  title={IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations
for LLM KV Cache Compression},
  author={Zhongping Ji},
  year={2026},
  note={Technical report}
}
```

## Next Steps

- Add real KV-cache extraction and evaluation on deployed LLMs
- Integrate a stage-2 residual correction path such as QJL
- Measure attention-logit fidelity, retrieval metrics, and perplexity
- Refine the paper into a full end-to-end conference submission
