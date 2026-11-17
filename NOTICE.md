# Notice: what is ours and what is the organisers'

This repository started as a copy of the organisers' public challenge repository,
<https://github.com/metrica-sports/more-than-a-hack-2027>, released under the MIT Licence
(`LICENSE`, © Metrica Sports). Those files are kept unchanged, apart from the README, which was
renamed. The match data is **not** in this repository and is not covered by that licence.

## From the organisers (MIT, Metrica Sports)

| Path | What |
|---|---|
| `README_ORGANISERS.md` | the organisers' `README.md`, renamed |
| `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `OVERVIEW.md`, `RULES.md`, `SUBMISSION.md` | challenge documentation |
| `LICENSE`, `CITATION.cff`, `.gitignore` | licence, citation, ignore rules |
| `games.py` | per-match constants, asset paths, frame alignment |
| `evaluation/` | official scorer, validator, metric library, public 5-minute ground truth, baselines |
| `starter/` | starter loaders, baseline and notebooks |
| `data/README.md` | where to unpack the data |
| `requirements.txt`, `requirements-starter.txt` | the organisers' dependency lists |

## By team Novara

| Path | What |
|---|---|
| `solution/` | the method that produces the submission |
| `training/` | the other-game experiment (Metrica public sample data), not used for the submission |
| `TECHNICAL_REPORT.md` | technical document |
| `README.md`, `NOTICE.md` | this documentation |
| `requirements-solution.txt` | pinned environment for the solution |

Our own code is released under the same MIT terms.
