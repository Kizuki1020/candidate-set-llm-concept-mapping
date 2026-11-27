Scientific Reports revision | Data and code accompanying the revision | Final, 27 November 2026
GitHub repository: https://github.com/Kizuki1020/candidate-set-llm-concept-mapping
Repository DOI: [DOI to be inserted after upload]
Reviewer access link: [reviewer access link to be inserted after upload]
The authors have confirmed that the materials in this package may be redistributed for use with this article. A specific open license has not been assigned in this package.

Work from the extracted directory. Frozen-output checks require Python, numpy, pandas and scipy; plotting also requires matplotlib. See REQUIREMENTS_VERIFIED.txt for the historical verification environment. Both datasets have 150 records and three runs per condition. Match by record identifier. The three repetitions are not independent observations.

  python3 verification/verify_package.py
  python3 verification/check_reference_and_lists.py
  python3 verification/p0c_analysis.py
  python3 verification/make_fig3.py
  python3 verification/make_fig4.py
  python3 verification/revision_analysis.py . revision_outputs

The original 53-check verifier retains its historical labels. Its seven “correct rejection” checks count exact string matches on a broad substring filter. For manuscript interpretation use revision_outputs/revision_checks.json and Supplementary Table S13. Four of the seven use the exact explicit label; the two definitions are not interchangeable. The revision script generates the updated paired tests, Holm correction, mass statistics, regression, and Figure 5. Legacy 123-row tables are included only to identify the discrepancy in the earlier version. They must not be pooled with the 150-record formal analysis.

diagnostic/ contains the two reference fields, raw outputs for the three commercial models, and archived diagnostic figure summaries. The Qwen four-condition outputs in results/exp1/ are a separate task. The Qwen diagnostic summary is included; a full per-record Qwen diagnostic export has not been located for this package. Older diagnostic CSVs omit a stable cross-model identifier. For Figure 2c, join by term, L class, gold MSC, expert and ontology labels, then within-key occurrence order. In one duplicate pair the pairing changes the triple-code overlap by one. See manuscript Results.

The original inference and data-construction programs are included as provenance. Historical script filenames, import layouts, local paths, software versions, source inputs and hash-based permutations may require reconstruction before retraining or inference can be attempted. The tested CPU workflow starts from archived predictions.

The manifest hashes the distributed files before any scripts are run. From the extracted root use shasum -a 256 -c MANIFEST.sha256. No rights to third-party material are asserted.

27 November 2026 revision to the local author-review package
The core inference outputs are unchanged. Additional CPU commands:
  python3 verification/audit_candidate_composition.py
  python3 verification/replay_construction.py

composition_audit/allowed_strings.csv gives one allowed string per domain, record, condition and member, in Unicode sort order. UC has no finite list. HC is derived from stored candidates. RO and SC are reconstructed using the archived build_pools/build_sets functions, seed 20260101 reset per domain, and the archived JSON order for one uninterrupted pass. The original per-run shuffle order, resume history and termination flags were not archived. Of 2,700 raw constrained outputs, 2,676 are exact members and the other 24 are unique proper prefixes. The CSV is not a log of the original order. Primary scores are frozen. The separate sensitivity analysis below rescored hypothetical completions. The other CSVs and audit.json contain the rowwise output comparison, length measures, resource provenance, literal corpus overlap counts, and both gain correlations and the additional RO-HC regression.

construction_audit/input_160.json is the recovered 160-record construction input restricted to fields used for the fixed-ontology replay. This file is distinct from the formal 150-row probe. The 160 pre-injection candidate lists themselves are archived in construction_audit/pre_injection_lists_160.json and pre_injection_lists_160.csv, so the permutation calculation does not depend on rebuilding the candidate pool. The two permutation coverage values in replay_comparison.json differ: 36/160 in the historical dataset metadata and 28/160 from the archived lists. Reference coverage before injection is 156/160, with four later injection flags. The observed 8-record difference in permuted coverage lies inside the permutation distribution reported below, so it is consistent with ordinary sampling variation; it is not attributed to any specific historical cause. No new model inference was performed.

Final revision: three additional CPU verification scripts
  python3 verification/permutation_distribution.py
  python3 verification/prefix_sensitivity.py
  python3 verification/final_consistency.py

The permutation script loads the archived pre-injection lists in construction_audit/pre_injection_lists_160.json (also written as pre_injection_lists_160.csv) and does not rebuild the candidate pool. A 160-by-160 canonical membership matrix retains repeated labels at their observed frequencies. It applies the archived builder's Python random.Random(20260101) single shuffle to those fixed lists (28 hits) and samples 100,000 uniform permutations without replacement within each draw (NumPy PCG64, seed 20261127). Results in construction_audit/permutation_distribution.json and permutation_frequencies.csv report hits per 160 records. Archived metadata report 36 hits. Both counts lie in the central discrete 95% range, 19–36. This does not establish the historical cause of the discrepancy. The lists were frozen under Python 3.9.6 and NumPy 2.0.2, which is the environment that reproduces the manuscript figures. Changing PYTHONHASHSEED within that environment did not change the lists. A live rebuild under a different Python or NumPy version can change list membership and is not used for the reported distribution. The archival field structural_lift subtracts permuted pre-injection coverage from true post-injection coverage. It includes four forced reference insertions and upstream test-label dependence. Interpret it only as a descriptive coverage difference. The builder source and its original console wording are retained for provenance.

The prefix script joins frozen predictions to reconstructed memberships, checks every primary score with the archived parser/matcher, and lists the 24 non-member outputs and their unique completions in composition_audit/prefix_runs.csv. It reports original scoring, hypothetical unique completion and complete-record exclusion in prefix_sensitivity.json (accuracy by condition, paired RO-HC and RO-SC gains, discordant counts and unadjusted two-sided exact McNemar p). Three runs give one majority vote per record and condition. No inferred completion replaces an archived output or establishes the reason generation stopped.

verification/final_consistency.py re-derives every quantity quoted in the manuscript and supplement from the frozen files in this archive and prints one PASS/FAIL line per claim (68 checks). It is the quickest way to confirm that the text and the data agree.

In the final frozen-output check, all twelve verification scripts listed in this README were run again after extraction. Set PYTHONHASHSEED=0, PYTHONDONTWRITEBYTECODE=1, MPLBACKEND=Agg and SOURCE_DATE_EPOCH=1795737600 for deterministic re-export of generated files in the tested environment; point MPLCONFIGDIR to a writable temporary directory. Alternate runtime versions can change plot serialization. For reference, the main diagnostic Qwen results are archived summaries only. The three commercial diagnostic files lack a shared permanent record ID. The figure summary inputs and non-equivalent subset diagnostics should be interpreted with these limitations.

  python3 verification/make_fig2.py
Figure 2 uses all 181 ontology-NONE records for each commercial model and the explicitly identified archived Qwen summary. The historic fig2_L3_confidence_by_outcome.csv used smaller commercial subsets and is retained for provenance only. For the revised Figure 2 use verification/figure2_source.csv and figure2_overlap.csv.
