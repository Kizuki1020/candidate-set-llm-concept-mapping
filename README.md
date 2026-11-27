# Candidate-set composition in constrained LLM concept mapping

Data and code accompanying the Scientific Reports revision of *Candidate-set composition shapes reference-label selection in constrained LLM concept mapping*.

- GitHub repository: https://github.com/Kizuki1020/candidate-set-llm-concept-mapping
- Repository DOI: `[DOI to be inserted after upload]`
- Reviewer access link: `[reviewer access link to be inserted after upload]`

The authors have confirmed that the materials in this package may be redistributed for use with this article. A specific open license has not been assigned in this package.

Full documentation, verification commands, and file descriptions are in [`README.txt`](README.txt). After extracting or cloning this repository, run the checks from the repository root, for example:

```bash
shasum -a 256 -c MANIFEST.sha256
python3 verification/final_consistency.py
python3 verification/permutation_distribution.py
python3 verification/replay_construction.py
```

`verification/permutation_distribution.py` loads the archived 160 pre-injection candidate lists and does not rebuild the candidate pool. The lists were frozen under Python 3.9.6 and NumPy 2.0.2; do not replace them with a live rebuild from another Python or NumPy version.

The Research Square preprint DOI `10.21203/rs.3.rs-10813874/v1` is not the repository DOI for this package.
