# METU Risk Persona

**Multi-Risk BART + DOSPERT-30 Behavioural Assessment Platform**

An open-source behavioural assessment platform that combines a multi-colour Balloon Analogue Risk Task (BART) with DOSPERT-30 self-report data, designed for research on risk-taking and decision-making. The platform captures fine-grained behavioural signals from gamified tasks and maps them onto interpretable risk profiles using unsupervised clustering. It was developed as part of an undergraduate honors seminar at Middle East Technical University (METU).

---

## How It Works

### Three-Colour BART

The task presents participants with 30 balloons (10 per colour, shuffled) drawn from three risk tiers:

| Colour | Max Pumps | Risk Tier | EV-Optimal Stop | Peak EV |
|--------|-----------|-----------|-----------------|---------|
| Purple | 128       | Low       | 11 pumps        | 6.46    |
| Teal   | 32        | Medium    | 5 pumps         | 3.04    |
| Orange | 8         | High      | 2 pumps         | 1.31    |

Neutral colours are used deliberately to avoid psychological bias (e.g., red = danger).

### Explosion Model

Each pump attempt *k* is an independent Bernoulli trial with linearly increasing probability:

```
P(explode at pump k) = k / maxPumps
```

This is a sequential model, not a pre-drawn uniform threshold. The EV-optimal stopping point is found by maximising `EV(s, N) = s * product(1 - k/N, k=1..s)` over integer `s`. A continuous approximation gives `sqrt(N)`, but the exact discrete peaks are 11/5/2 for our three colours.

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
│       ├── BartGame.tsx                React/Next.js game client component
│       └── ResearchThankYou.tsx        Post-session risk profile display
├── scoring/
│   ├── bart.py                         EV-based BART scoring engine
│   └── schemas/
│       └── game_events.py              Pydantic event schemas and validators
├── clustering/
│   ├── clustering_pipeline.py          K-Means clustering with PCA visualization
│   └── clustering_results/
│       ├── clustered_participants.csv  Cluster assignments (N=10)
│       ├── summary.txt                 Silhouette scores and cluster profiles
│       ├── 01_k_selection.png          Elbow + silhouette plots
│       ├── 02_cluster_profiles.png     Z-scored cluster centroids
│       └── 03_pca_projection.png       PCA scatter plot
├── docs/
│   ├── bart_metrics_reference.tex      Full technical reference (all metrics)
│   └── figures/
│       ├── 01_k_selection.png
│       ├── 02_cluster_profiles.png
│       └── 03_pca_projection.png
├── data/
│   └── synthetic/
│       └── .gitkeep                    Real participant data is never committed
└── scripts/
    └── generate_synthetic.py           Generate synthetic demo datasets
```

---

## Key Metrics

The scoring engine computes EV-based calibration metrics, a composite impulsivity index, and money efficiency among others. The clustering pipeline uses three features:

| Feature | Source | Description |
|---------|--------|-------------|
| `dospert_financial` | DOSPERT-30 | Self-reported financial risk tolerance. |
| `rng_normalized_pumps` | BART | Mean collected pumps normalised by each colour's maxPumps. Measures behavioural risk appetite while controlling for colour difficulty. |
| `impulsivity_index` | BART | Composite of timing impulsivity (40%), excess explosions over EV-optimal baseline (40%), and orange signal (20%). Always computable regardless of collection counts. |

Clustering runs on N=10 (prior-task participants excluded), yielding k=3 with silhouette = 0.475. See `docs/bart_metrics_reference.tex` for the full technical reference covering all metrics, composites, and archetype assignment logic.

---

## Tech Stack

- **Game client:** Next.js / React (TypeScript)
- **Scoring engine:** Python, NumPy, SciPy
- **Clustering:** scikit-learn (K-Means, PCA, silhouette analysis)
- **Schemas:** Pydantic (event validation and type safety)

---

## References

* Blais, A.-R. & Weber, E.U. (2006). A Domain-Specific Risk-Taking (DOSPERT) scale for adult populations. Judgment and Decision Making, 1(2), 33–47.
* Dinç, S.C. & Yavaş Tez, Ö. (2019). Alana özgü risk alma ölçeği — kısa formu’nun (DOSPERT-30) Türkçeye uyarlama çalışması [The adaptation study into Turkish of the Domain-Specific Risk-Taking Scale — Short Form (DOSPERT-30)]. Spor Bilimleri Dergisi, 30(3), 107–120. https://doi.org/10.17644/sbd.471304
* Lejuez, C.W., Read, J.P., Kahler, C.W., et al. (2002). Evaluation of a behavioral measure of risk taking: The Balloon Analogue Risk Task (BART). Journal of Experimental Psychology: Applied, 8(2), 75–84.
* Weber, E.U., Blais, A.-R., & Betz, N.E. (2002). A domain-specific risk-attitude scale: Measuring risk perceptions and risk behaviors. Journal of Behavioral Decision Making, 15(4), 263–290.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{meturiskpersona2026,
  author    = {Yılmaz, Ahmet Selim},
  title     = {METU Risk Persona: A Multi-Risk BART Platform for Behavioural Assessment},
  year      = {2026},
  url       = {https://github.com/aslmylmz/metu-risk-persona},
  note      = {Open-source software}
}
```
