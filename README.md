# QATNN-HFP — Paper README

Reader's guide to **QATNN-HFP-EN.pdf**: *Quantized Adaptive Trainer Neural Network – Hybrid Fixed Point* — a ternary, integer-only training and inference regime in formal academic English (translated from Italian).

## What this paper is

> Unlike ternary engines that impose quantization as an afterthought, QATNN makes the trit the unit of account of the whole matrix traversal — forward and backward in integer arithmetic only — and confines floating point to vector normalizations; thus it obtains multiplier-free inference at ~1.6 bits per weight without ternary training freezing.

- **Thesis:** quantization designed as the arithmetic regime of the engine, not as post-hoc compression.
- **Scope:** bit-exact spec of every operator (§1–§9), threshold-reachability analysis (§7), honest split between verified core and `[MIO]`-marked extensions.
- **Status:** English translation of the author's Italian draft. Formulas, code tokens (`__builtin_clz`, `>>`/`<<`, `int64_t sum_abs = 0;`), variable names, numeric values and all citations preserved character-for-character. Only Italian prose *inside* equation boxes was rendered into English. Tight source spacing (e.g. `output:If γshift…`) kept for fidelity.

## File

- `QATNN-HFP-EN.pdf` — PDF file, 196 paragraphs in original order, same Title/Heading styles, bold/italic/code emphasis, 182 Office-Math equations byte-identical, both figures, 38 hyperlinks, 34 bookmarks, headers/footers intact.

## How to read it (10 sections)

1. **Base architecture + forward pass** — ternary weights {-1,0,+1}, adds/subs replace MACs, per-layer `decay`, `γraw=(Σ|shadow|)≫(31−clz(N))`, `γshift` pure-shift scaling, **SiLUNS** division-free SiLU (`x≥0→x`, `0>x>−41900→x≫2`, `x≤−41900→−(x≫2)−20950`, `C0`-continuous, `O(1)`), 1.625 bit/weight packing (≈3.25 MiB at 4096×4096, −80%).
2. **Hysteresis map** — Shadow(Q16.15)→trit with `T=213`, asymmetric transitions, uniform trit init + Gaussian shadow init, STE unlocks zero trits, SIMD keeps zero-skip omitted in forward.
3. **Chunks + memory** — `I_X.bin` [Input][Trit][Output][decay][gamma_shift] + `S_X.bin` [Shadow][decay Weight], selective mmap, `I_{X+1}` transposed-trit rule, async double-buffered prefetch, max 2 layers resident.
4. **Formats + HFP contract** — Shadow Q16.15, Output Q16.15, Grad Q4.11, LR UQ0.16, Acc_Err UQ7.25, Acc_Grad int64, `γshift` int8. Zero FPU in matrix path; FP32 only for RMSNorm, RWKV state, Softmax/Cross-Entropy.
4.1/4.2 **RWKV language integration** — no attention / spatial softmax; RWKV-6 Finch linear recurrence + Pre-Norm + SwiGLU (`ff=SiLUNS(Wg·x)·(Wu·x)`), ternary embedding + BPE.
5. **Error + gradient** — `ErrorL=ŷ−y`, `∂Loss/∂Output=2(Output−Target)`, `ErrorX=Tritᵀ_{X+1}×Error_{X+1}`, `GX=(Error_{X+1}⊗InputX)≫19` clipped to [−2047,+2047], rank-1 row-by-row (14 GB at 7B, not 26 GB — 26 GB holds only past 13B params), int64 Q32.30 staging.
6. **Decay (UQ4.12)** — `decayWeight` init from `0.5·log2(Nin)`, `Gdecay=((Σ|E·X|)≫8)&0xFFFF`, `Δ=(G·LR)≫15`, `decay=decayWeight≫12` clipped 0–15.
7. **Update + 4 levers** — `ΔShadow=(G·LR)≫(12−Update_Gain)`, gain default 0; grad-scale 8-step average, σ=2048 init, WSD warmup-10 + stall-halving. Reachability: T=213 → min |G|=14 at max LR, 96.0% live pairs; T=8192 → 513, 40.3%.
8. **8-inference cycle** — 7× accumulate Acc_Err/Acc_Grad, 8th: `GAvg≫3`, `LR=max(64,min(65535,|GAvg|+(AccErr≫(9+Cooling))))`, floor 64, Cooling+Shift grows the divisor.
9. **Layer cut** — `min(tot,max(40,tot),42949672)×p100`, 40 = compact-model floor, integer truncation only.
10. **References** — BitNet b1.58, RWKV, integer-arithmetic inference (Jacob et al.), Sparse-BitNet / TernaryLM 2026 preprints, BPE, SiLU, HPC I/O overlap.

## Reproducibility / verification notes

- 182/182 equations byte-identical to source; zero Italian words or accented characters remaining in prose (citations excluded).
- Package check: identical paragraph order/styles, media, relationships, links and bookmarks vs source.
- Reachability table (§7) is analytic derivation from the stated update rule, not multi-seed measured means — flagged as future work in the paper.
- BITCOS ≥51%-zero probe (`qat_fit_bitcos`) exists; dynamic format switch not implemented.
- Italian normative companion (code-aligned r1): `v003/QATNN-HFP-r1.md`.

## Cite

QATNN-HFP (EN). *Quantized Adaptive Trainer Neural Network – Hybrid Fixed Point.* English translation of author's Italian manuscript. File: `QATNN-HFP-EN.pdf`.

Key external refs: Ma et al. BitNet b1.58 (arXiv:2402.17764), BitNet 2B4T (arXiv:2504.12285), Peng et al. RWKV (arXiv:2305.13048), Jacob et al. integer-arithmetic-only inference (arXiv:1712.05877), Sennrich et al. BPE (arXiv:1508.07909), Elfwing et al. SiLU (arXiv:1702.03118).
