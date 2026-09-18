# NonLinearRNNsCanBeParallel

MLP-RNN, rKAN-RNN, M²RNN, minGRU, and minLSTM implementations with sorted deterministic graph connectivity benchmark.

**Papers implemented:**
- **MLP-RNN**: arXiv:2603.03612 — "Nonlinear RNNs Can Be Parallel" (Definition 3, 7, 8, 9, 10; Theorem 1, Corollary 2)
- **rKAN-RNN**: arXiv:2406.14495v1 — "Rational Kolmogorov-Arnold Networks" (Section 2, 3; Eq. 9, 12)
- **M²RNN**: arXiv:2603.14360 — "Matrix-to-Matrix RNN" (Section 3.1, Eqs. 10–23)
- **minGRU/minLSTM**: arXiv:2410.01201v3 — "Were RNNs All We Needed?" (Sections 3.1–3.2, Appendix B)

Uses RMSNorm and SiLU activations, float32 precision.

## Quick start

```bash
uv sync --extra cpu --extra dev
uv run python scripts/train.py --fast-dev-run true
uv run python scripts/eval.py --fast-dev-run true
```

## Task

Sorted deterministic graph connectivity benchmark (Definition 11, arXiv:2603.03612): given source s, edges (i, j) in unary topological order, and target t, predict 1 iff there is a path s to t.

## Verification

```bash
uv run pytest tests/
```

## Citations

```bibtex
@misc{merrill2026nonlinearrnns,
  title         = {Nonlinear RNNs Can Be Parallel},
  author        = {Merrill, William and Jiang, Hongjian and Li, Yanhong and Lin, Anthony and Sabharwal, Ashish},
  year          = {2026},
  eprint        = {2603.03612},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{aghaei2024rkan,
  title         = {rKAN: Rational Kolmogorov-Arnold Networks},
  author        = {Aghaei, Alireza Afzal and Hosseinzadeh, Mehdi and Parand, Kourosh},
  year          = {2024},
  eprint        = {2406.14495},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{mishra2026m2rnn,
  title         = {M²RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling},
  author        = {Mishra, Mayank and Tan, Shawn and Stoica, Ion and Gonzalez, Joseph E. and Dao, Tri},
  year          = {2026},
  eprint        = {2603.14360},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}

@misc{feng2024rnns,
  title         = {Were RNNs All We Needed?},
  author        = {Feng, Leo and Tung, Frederick and Ahmed, Mohamed Osama and Bengio, Yoshua and Hajimirsadeghi, Hossein},
  year          = {2024},
  eprint        = {2410.01201},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}
```
