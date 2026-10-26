# QATNN-HFP: a ternary neural engine in hybrid fixed point with linear recurrence

**Revision r1 - aligned to v003 code.** Every formula reported is the one implemented; where the code admits multiple modes, the default mode is declared. Corrections with respect to the previous draft are integrated in Sections 3-4 and summarized in Section 8.

> **Thesis.** Unlike ternary engines that impose quantization as an after-the-fact constraint, QATNN makes the trit the unit of account of the entire matrix traversal - forward and backward in integer-only arithmetic - and confines floating point to vector normalizations only; in this way it obtains multiplier-free inference at 1.6 bits per weight without ternary training freezing.

## Abstract

QATNN (Quantized Adaptive Trainer Neural Network) is a training and inference engine in which weights take only values in {-1, 0, +1} (trits), governed by continuous shadow weights and a hysteresis map with threshold T = 213. The forward pass accumulates in Q16.15 with inputs pre-scaled by the layer decay and scales the output with an integer gamma-shift computed via `__builtin_clz`; the SiLUNS activation approximates SiLU with shifts and linear branches only. The backward pass never materializes the gradient matrix (rank-1 outer products evaluated row by row), propagates the error through transposed trits, and updates the shadows with integer steps. State goes to disk in per-layer I/S chunks with at most 2 layers resident in RAM and asynchronous prefetch. The language variant replaces attention with RWKV-6/Finch linear recurrence, SwiGLU blocks, RMSNorm in fp32 and character-level BPE (V = 1024). Four convergence levers - grad-scale with 8-step moving average, default Update_Gain 0, Gaussian init sigma = 2048, WSD schedule with stall-halving - bring the update step into the range where the hysteresis threshold can be crossed: with the stated rule 96.0% of (|G|, LR) pairs move a trit at T = 213 versus 40.3% at T = 8192 (derived analytically, Section 7).

## 1. Introduction

The phenomenon is well known: moving a network to low precision after training degrades quality exactly where the network is most sensitive, and recovery requires retraining in a format the hardware does not execute natively. Our conclusion is that quantization must be designed as the arithmetic regime of the engine, not as after-the-fact compression. The gap is that existing ternary regimes (BitNet b1.58 [Ma et al., 2024]) describe inference but leave the update in floating point. This work proposes an engine in which the update is also integer: shadows in Q16.15, gradients in Q4.11, learning rate in UQ0.16, and floating point only outside the matrix path (Section 4.1). We promise: (a) the bit-exact specification of every operator; (b) the measurement of hysteresis-threshold reachability; (c) the honest separation between verified core and extensions marked [MIO].

## 2. Related work

**Integer quantized training.** Jacob et al. (arXiv 2017; CVPR 2018) set the theoretical basis of integer-only inference; QATNN inherits its Q formats but replaces pure integer inference with a ternary shadow-weight and hysteresis regime. **Ternary networks.** BitNet b1.58 [Ma et al., 2024] and the 2B4T technical report [Ma et al., 2025] demonstrate 1.58-bit LLMs; the 2026 works Sparse-BitNet (Zhang et al., arXiv:2603.05168, exact subtitle: "1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity") and TernaryLM (Nargund and Shukla, arXiv:2602.07374, "Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling") extend the line to structured sparsity and per-layer adaptive scaling - both preprints with no venue, cited as such. These works do not train in an integer regime: that is our space. **Linear recurrence.** RWKV (Peng et al., Findings of EMNLP 2023; arXiv:2305.13048) replaces attention with O(T) recurrence; RWKVQuant (Xu et al., ICML 2025, PMLR vol. 267, pp. 70047-70064; arXiv:2505.03803) studies its PTQ quantization with hybrid scalar/vector guidance. QATNN adopts its recurrence but trains ternary weights natively. **Subwords and activations.** BPE [Sennrich et al., arXiv 2015; ACL 2016] and SiLU [Elfwing et al., arXiv 2017; Neural Networks 2018] are the two external bases: the former for vocabulary, the latter as the reference approximated by SiLUNS.

## 3. Method

### 3.1 Gamma-shift with hardware instructions (Section 1 of the core)

For each layer, `gammraw = (sum|shadow|) >> floor(log2(N_in))`, with int64 sum and N_in equal to the block size (a power of two in supported configurations; the `N_in == 0` guard avoids `__builtin_clz(0)`, undefined in C). The shift is derived as `gshift = 16 - floor(log2(gammraw))` (implemented via `(32 - __builtin_clz(g)) - 1`), in int8. Application is pure shift on the accumulator, with a guard against shifts >= 32 (on x86 the count is masked to 5 bits: `<< 47` would silently become `<< 15`, worse than a zero). Correction note: the previous draft reported the formula with a negative sign in front of `__builtin_clz`; the code implements the above and the two `r7`/`correct` variants coincide.

### 3.2 SiLUNS r7 (Section 1b)

In Q16.15, with minimum point at x = -41900 (about -1.28): x >= 0 -> x; 0 > x > -41900 -> x >> 2; x <= -41900 -> -(x >> 2) - 20950. Continuity at the boundary: -(-41900 >> 2) - 20950 = 10475 - 20950 = -10475, equal to the descending branch (-41900 >> 2 = -10475, about -0.319, versus the true SiLU minimum at -0.278). O(1) cost with no exponentials or divisions.

### 3.3 Hysteresis, T = 213 (Section 2)

The map is: above +T, prev -1 -> 0 (no direct jump to +1: intended asymmetry); below -T, prev +1 -> 0; inside (-T, +T) the trit holds. At startup trits are uniform over {-1, 0, +1} (no "previous state") and shadows from Gaussian sigma = 2048 (Section 3.7). Measured threshold reachability with `delta = (|G|*LR) >> 12`, |G| <= 2047, LR <= 65535: minimum |G| moving a trit at maximum LR = 14 for T = 213 versus 513 for T = 8192; live pairs on grid 96.0% versus 40.3% (derived, Section 7). The value 8192 was never the threshold: it appears only as a diagnostic comparison point (0.25 in Q16.15, 38.5x the true threshold).

### 3.4 Chunks, packing, prefetch (Sections 3-3.1)

Each layer persists as a pair I (inference: header with decay/gshift, trits, input/output) + S (optimizer: shadows, decay_w); the backward of layer X also opens I_{X+1} for transposed trits. Three distinct formats, not to be confused: (a) disk/RAM streaming = 1 byte per trit (chosen for SIMD and simplicity); (b) block archival of 128 weights in 26 bytes = 1.625 bits/weight (`qat_packed_bytes`); (c) flat `ceil(N/5)` = 1.600 bits/weight (`qat_repack`). At 4096x4096: 16.00 MiB -> 3.25 MiB in format (b), 3.20 MiB in format (c). Resident constraint: at most 2 layers in RAM with successor prefetch on a secondary thread and kill-switch `QATNN_NO_PREFETCH=1`.

### 3.5 HFP numerical contract (Sections 4-4.1)

Shadow Q16.15 int32; gradient Q4.11 int16 in [-2047, +2047] with clip; LR UQ0.16 with floor 64; decay_w UQ4.12; gamma int8. Zero FPU in the matrix traversal; fp32 only for RMSNorm, RWKV recurrent state and Softmax/Cross-Entropy loss. Implementation detail: the layer accumulator is int32 Q16.15 and after gamma+SiLUNS is kept in float for the [MIO] chain - the literal chain of the paper skips the added derivatives.

### 3.6 Language extension [MIO] (Section 4.2)

Pre-Norm block + RWKV-6/Finch O(T) time-mixing with fixed mix coefficients (never learned), SwiGLU MLP, ternary embedding with character BPE V = 256 + 768 = 1024. No attention, no spatial softmax. SiLUNS, RMSNorm and gate derivatives do not belong to the literal chain and are marked in the code.

### 3.7 Backward and update (Sections 5-7)

Output error e = 2*(y_hat - y) in Q16.15 (the ">> 19" of the loss concerns printing only); layer error E_X = E_{X+1} * trit^T(X+1); gradient G = (E_{X+1}*X_q) >> 19 in int64 with Q4.11 clip, never materialized (row by row). Decay: gacc = sum|ev|*|xs| with xs = x_q >> decay; G_decay = (gacc >> 8) & 0xFFFF; delta = (G_decay*LR) >> 15; decay_w -= delta; decay = clip(decay_w >> 12, 0, 15). Engine decay init: round(0.5*log2(n_in)), clipped (coincides with min(15, (31-clz)>>1) << 12 for even log2; the core uses fixed default 3). Shadow: delta = (q*LR) >> (12 - Update_Gain), default gain 0. The four levers: (1) grad-scale with 8-step moving average bringing the gradient back to absolute scale before the product with the input (without it: everything saturates at +-2047 and only the sign remains); (2) Update_Gain reducing the shift; (3) sigma = 2048, about 9.6 thresholds - high value, declared as such: breaks symmetry but is born "decided by noise"; (4) WSD with automatic 10-step warmup (default on, calibrated for resuming from existing weights) and stall-halving (halves after 3 worsening windows) [MIO]; the global WSD scheduler is default off. Adaptive window: default constant LR with optional recompute every 8 steps; recompute uses (ag >> s) + (ae >> s), s = 9 + Cooling_Shift - the draft formula with unshifted |G_Avg| does not match the implemented default. Honest note: with UQ7.25 loss the error term saturates the 65535 ceiling, so the engine re-clips to LR_MAX = 4096 and forbids climbing back once the ceiling is reached. Layer cut: integer C with minimum threshold 40 and cap 42949672 (the UINT32 overflow of the intermediate product is known and contained by clamping to the total).

## 4. Results

**Threshold reachability (derived analytically from the stated rule).** T = 213: min |G| at max LR 14; live pairs 96.0%. T = 8192: min |G| 513; live pairs 40.3%. At typical gradient |G| = 7 even T = 213 is unreachable without grad-scale (max 111 < 213): lever (1) is necessary, not optional. **Convergence.** The threshold scale confirms 213 lives and 8192 freezes; sigma/T = 9.62 versus 0.25 confirms the inverted init in the second case. Stated limit: these are derivations from the update rule, not means over >=5 seeds with intervals - training measurements remain future work (Section 5).

## 5. Discussion and limits

Technique limit: no softmax attention (choice: RWKV); no continuous cooling schedule until implemented; 8-window with dominant I/O on slow disk (masked by prefetch, not eliminated). Experiment limit: one dataset, small n, one language; tables without error bars because derived, not measured. The BITCOS threshold (>=51% zeros, measured by `qat_fit_bitcos`) exists as a probe; dynamic format switching is not implemented.

## 6. Conclusions

The verifiable contribution is the complete integer regime - forward, row-by-row backward and update - with the threshold-reachability measurement and four declared levers. Open remains: verified continuous schedule, multi-seed measurements with intervals, and BITCOS activation above threshold.

## 7. References

1. Sennrich, R., Haddow, B. and Birch, A. (arXiv 2015; ACL 2016, pp. 1715-1725). Neural Machine Translation of Rare Words with Subword Units. arXiv:1508.07909. DOI: 10.18653/v1/P16-1162.
2. Elfwing, S., Uchibe, E. and Doya, K. (arXiv 2017; Neural Networks, vol. 107, 2018, pp. 3-11). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. arXiv:1702.03118. DOI: 10.1016/j.neunet.2017.12.012.
3. Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764. DOI: 10.48550/arXiv.2402.17764.
4. Ma, S. et al. (2025). BitNet b1.58 2B4T Technical Report. arXiv:2504.12285. DOI: 10.48550/arXiv.2504.12285.
5. Peng, B. et al. (Findings of EMNLP 2023, pp. 14048-14077). RWKV: Reinventing RNNs for the Transformer Era. arXiv:2305.13048. DOI: 10.18653/v1/2023.findings-emnlp.936.
6. Xu, C. et al. (ICML 2025, PMLR vol. 267, pp. 70047-70064). RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization. arXiv:2505.03803. OpenReview: UOw6Qt0qYU.
7. Zhang, D. et al. (2026, preprint). Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity. arXiv:2603.05168.
8. Nargund, N. and Shukla, P. (2026, preprint). TernaryLM: Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling. arXiv:2602.07374.
9. Jacob, B. et al. (arXiv 2017; CVPR 2018, pp. 2704-2713). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. arXiv:1712.05877. DOI: 10.1109/CVPR.2018.00286.

## 8. Pre-submission check

Thesis in one sentence: yes. Numbers with error/interval: partial - the tables in Section 4 are analytic derivations declared as such; training measurements are missing (Section 5). Baseline with reference: yes for external formats (BitNet). Claim with ablation: partial (levers described, future ablations). Technique/experiment limits: yes. Entries read or verified via primary source: yes for all (arXiv/ACL/DOI above); entries 7-8 are explicitly marked preprints. Abstract written last: yes.
