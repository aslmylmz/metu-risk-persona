# METU Risk Persona

**Multi-Risk BART + DOSPERT-30 Behavioural Assessment Platform**

An open-source behavioural assessment platform that combines a multi-colour Balloon Analogue Risk Task (BART) with DOSPERT-30 self-report data, designed for research on risk-taking and decision-making. The platform captures fine-grained behavioural signals from gamified tasks and maps them onto interpretable risk profiles using unsupervised clustering. It was developed as part of an undergraduate honors seminar at Middle East Technical University (METU).

---

## How It Works

### Three-Colour BART

The task presents participants with 30 balloons (10 per colour, shuffled) drawn from three risk tiers:

| Colour | Max Pumps | Risk Tier | Optimal Stop (approx.) |
|--------|-----------|-----------|------------------------|
| Purple | 128       | Low       | ~32 pumps              |
| Teal   | 32        | Medium    | ~8 pumps               |
| Orange | 8         | High      | ~2 pumps               |

Neutral colours are used deliberately to avoid psychological bias (e.g., red = danger).

### Explosion Model

Each pump attempt *k* is an independent Bernoulli trial with linearly increasing probability:

```
P(explode at pump k) = k / maxPumps
```

This is a sequential model, not a pre-drawn uniform threshold. The expected-value-maximising stopping point under this model is approximately `maxPumps / 4`, which differs from the `maxPumps / 2` heuristic of the classic uniform BART.

### Session Structure

A session consists of 30 balloons (10 per colour, presented in shuffled order). On each balloon the participant repeatedly chooses to pump (inflating the balloon and increasing the potential reward) or to collect (banking the reward). If the balloon explodes, the reward for that balloon is lost.

---

## Repository Structure

```
metu-risk-persona/
├── README.md                           Project overview and documentation
├── LICENSE                             MIT License
├── .gitignore                          Git ignore rules
├── games/
│   └── bart/
│       └── BartGame.tsx                React/Next.js game client component
├── scoring/
│   ├── bart.py                         NumPy-vectorized BART scoring engine
│   └── schemas/
│       └── game_events.py              Pydantic event schemas and validators
├── clustering/
│   └── clustering_pipeline.py          K-Means clustering with PCA visualization
├── docs/
│   ├── bart_metrics_reference.tex      LaTeX reference for metric definitions
│   └── figures/
│       └── .gitkeep
├── data/
│   └── synthetic/
│       └── .gitkeep                    Real participant data is never committed
└── scripts/
    └── generate_synthetic.py           Generate synthetic demo datasets
```

---

## Key Metrics

The clustering pipeline operates on six behavioural features extracted from BART sessions:

| Feature | Description |
|---------|-------------|
| `rng_normalized_pumps` | Mean collected pumps normalised by each colour's maxPumps, averaged across colours. Measures overall risk appetite while controlling for colour difficulty. |
| `impulsivity_index` | Proportion of balloons where the participant pumped beyond the EV-optimal stopping point. Higher values indicate greater impulsivity. |
| `patience_index_normalized` | Mean time spent deliberating per pump, normalised to [0, 1]. Captures how cautiously participants approach each decision. |
| `mean_latency_between_pumps` | Average inter-pump interval in milliseconds. Reflects motor tempo and deliberation speed. |
| `between_balloon_consistency` | Coefficient of variation of pump counts across balloons. Lower values indicate a more consistent strategy. |
| `adaptive_strategy_score` | Degree to which participants differentiate their pumping behaviour across the three colour tiers. Higher values indicate better calibration to risk levels. |

---

## Tech Stack

- **Game client:** Next.js / React (TypeScript)
- **Scoring engine:** Python, NumPy, SciPy
- **Clustering:** scikit-learn (K-Means, PCA, silhouette analysis)
- **Schemas:** Pydantic (event validation and type safety)

---

## References

- Lejuez, C. W., Read, J. P., Kahler, C. W., Richards, J. B., Ramsey, S. E., Stuart, G. L., Strong, D. R., & Brown, R. A. (2002). Evaluation of a behavioral measure of risk taking: The Balloon Analogue Risk Task (BART). *Journal of Experimental Psychology: Applied*, 8(2), 75--84.

- Blais, A.-R., & Weber, E. U. (2006). A Domain-Specific Risk-Taking (DOSPERT) scale for adult populations. *Judgment and Decision Making*, 1(2), 33--47.

- Dinc, S.C., & Yavas Tez, Ö. (2019). Alana Ozgu Risk Alma Olcegi-Kisa Formu'nun (DOSPERT-30) Turkceye uyarlanmasi. *Spor Bilimleri Dergisi*, 30(3), 107--120. https://doi.org/10.17644/sbd.471304

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{meturiskpersona,
  author    = {Yılmaz, Ahmet Selim},
  title     = {METU Risk Persona: A Multi-Risk BART Platform for Behavioural Assessment},
  year      = {2026},
  url       = {https://github.com/aslmylmz/metu-risk-persona},
  note      = {Open-source software}
}
```
