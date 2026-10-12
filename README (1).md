# 🦵 RSNA Knee Abnormality Detection: the complete pipeline in one notebook

![Public LB](https://img.shields.io/badge/Public%20LB-0.949-brightgreen)
![Metric](https://img.shields.io/badge/metric-macro%20ROC--AUC-blue)
![Platform](https://img.shields.io/badge/Kaggle-GPU%20T4%20×2-20BEFF?logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)

An end-to-end Kaggle notebook that reads raw knee-MRI DICOM studies, scores **12 abnormalities** per study and writes `submission.csv`. It explains every image-processing and modelling decision on the way.

> **Public leaderboard: 0.949** (default settings)

The notebook does three jobs:

| | |
|---|---|
| 🏆 **Submit** | runs a fused CoAtNet + ConvNeXt ensemble and writes the scored file |
| 📚 **Teach** | walks one study from DICOM to tensor, takes each model apart, and runs a small training loop live |
| 🔍 **Audit** | measures label noise, tests what each preprocessing step is worth, and reports which models actually reached the published file |

---

## Contents

- [Quick start](#quick-start)
- [The task](#the-task)
- [Pipeline](#pipeline)
- [Notebook map](#notebook-map)
- [Settings](#settings)
- [Outputs](#outputs)
- [Results](#results)
- [Lessons learned](#lessons-learned)
- [Known limitations](#known-limitations)
- [Next steps](#next-steps)
- [Credits](#credits)

---

## Quick start

1. Import the notebook into Kaggle. The accelerator should be **GPU T4 ×2**, with internet **off**.
2. **Add Input → Datasets → `harshikasipani7/knee-model-weights`.** This dataset is needed for Branch S, and it's the one input that isn't attached automatically.
3. Click **Run All**. The notebook writes `submission.csv`.
4. Submit that version.

With the default settings, Kaggle's scoring run takes Parts A–D (a few minutes) plus Branch A.

---

## The task

For every knee MRI study, score 12 findings:

| Ligaments | Menisci | Osteoarthritis | Joint & bone |
|---|---|---|---|
| ACL tear | Medial meniscus tear | Medial OA | Effusion |
| MCL tear | Lateral meniscus tear | Lateral OA | Synovitis |
| | | Patellofemoral OA | Baker's cyst |
| | | | Bone contusion |
| | | | Fracture |

**Metric: macro ROC-AUC**, averaged over the 12 findings. It only looks at the *order* of the scores, so the pipeline blends **percentile ranks**, not probabilities.

**Why it's hard**
- **Few real labels.** Only 58 of the 4,407 training studies have official labels. The other 4,349 come with a free-text radiology report, and their labels were extracted by a language model. Those labels are noisy (macro AUC 0.870 against the official ones).
- **No two studies alike.** Scanners, pixel sizes (0.14–0.89 mm), fields of view (130–230 mm), and the number of series and slices all vary from study to study.

---

## Pipeline

```
                         ┌────────────────────────────────────────┐
                         │  Branch A  ·  scored by default         │
                         │  A1  CoAtNet · 96 slices · 94 windows   │
                         │      plain + anatomical mirror          │
                         │  A2  ConvNeXt 2.5D reader · 8× TTA      │──┐
                         │  A3  per-finding rank fusion            │  │
                         └────────────────────────────────────────┘  │
                         ┌────────────────────────────────────────┐  │   ┌─────────────┐
 DICOM ─► preprocessing ─┤  Branch B  ·  weight v (default 0)      │──┼──►│  Stage C    │──► submission.csv
                         │  DINOv2 ×20 → A5 ×5 → Raptor → CoAtNet  │  │   │  rank blend │
                         └────────────────────────────────────────┘  │   └─────────────┘
                         ┌────────────────────────────────────────┐  │
                         │  Branch S  ·  weight s (default 0)      │──┘
                         │  frozen DINOv2-small + our 5 MLP heads  │
                         └────────────────────────────────────────┘
```

**Final blend:** `rank( (1 − v − s)·rank(A) + v·rank(B) + s·rank(S) )`

<details>
<summary><b>Stage-by-stage detail</b></summary>

| Branch | Stage | Model | Input built from a study | Combined how |
|---|---|---|---|---|
| **A** | A1 | CoAtNet `coatnet_rmlp_2_rw_384`, 1 SWA checkpoint | 96 slices in 5 slots → 94 three-slice windows, 384 px (central 320 px used) | percentile rank |
| | A2 | ConvNeXt-Tiny 2.5D reader, 3 folds | 6 slots × 16 windows; 2 sizes × flip × 2 offsets | mean probability |
| | A3 | fusion | — | per finding: `(1−w)·rank(A1) + w·rank(A2)` |
| **B** | B1 | DINOv2 ViT-S/14 × 20 members | 6 slots × 12 slices, right knees mirrored | mean of 20 ranks |
| | B2 | A5 × 5 folds | 6 slots × 16 slices used as 16 channels | 52 % into B1 |
| | B4 | Raptor CoAtNet × 4 views | up to 96 slices at 384 px | weighted mean |
| | B5 | CoAtNet family × 4 readers | similar | 40 % into B4, then 60–100 % against B1–B2 |
| **S** | S | frozen DINOv2-small + 5 MLP fold heads (6,148 → 512 → 12) | 4 slots × 16 slices at 224 px, raw intensities | mean of 5 folds |

</details>

<details>
<summary><b>How each model family preprocesses a study</b></summary>

A network has to get exactly the preprocessing it was trained with, so each family keeps its own reader:

| Decision | A1 CoAtNet | A2 ConvNeXt | B1 DINOv2 | B2 A5 |
|---|---|---|---|---|
| Slice order | physical position | physical position, fixed direction | physical position | `InstanceNumber` |
| Slices used | 96 in 5 slots, 2–98 % of each series | ≤ 64 per series; 16 windows per slot | 12 per slot, central 20–80 % | 16 per slot, 12–88 % |
| Field of view | central 140 mm | 153.6 mm at 0.4 mm/px | central 130 mm | central 130 mm |
| Intensity window | 2nd–98th percentile | 1st–99.5th percentile | 1st–99th percentile | 1st–99th percentile per slice |
| Image size | 384 px → central 320 | 256 and 320 px | 336 px | 336 px |
| Left / right | plain + mirrored pass | plain + flipped pass | right knees mirrored | coronal/axial mirrored |
| Depth | 3 slices as RGB | 3 slices as RGB | 3 slices as RGB | 16 slices as 16 channels |

</details>

---

## Notebook map

| Part | What's inside |
|---|---|
| **A** | Settings, path resolution, helpers; what the competition data contains |
| **B** | Radiology-report text preprocessing: NFKC normalisation, script/language detection, keyword flags |
| **C** | Audit of the LLM pseudo-labels against the 58 official labels |
| **D** | Image processing, one decision per cell: DICOM header → slice order → mm crop → percentile window → orientation and laterality → 2.5D stacking → resize and normalise |
| **E** | The model pipeline: Branch B → Branch A → Branch S → Stage C |
| **F** | Training explained: checkpoint contents, augmentation, live training demo |
| **G** | Ablation: remove one preprocessing step at a time and measure the cost |
| **H** | Run health, submission validation, comparison of model outputs |
| **I** | Validation against official labels (validate mode only) |
| Appendix | Code provenance, limitations, ideas, glossary |

---

## Settings

All settings are in the first code cell.

| Setting | Default | Effect |
|---|---|---|
| `RUN_MODE` | `"submit"` | `"submit"` predicts the test set and writes `submission.csv`. `"validate"` runs the identical pipeline on the 58 labelled studies and scores every stage. |
| `FIRST_PIPELINE_WEIGHT` | `0` | Branch B's share (`v`) of the final blend |
| `OWN_MODEL_WEIGHT` | `0` | Branch S's share (`s`) of the final blend |
| `RUN_DEMOS` | `True` | `False` skips the training demo and the ablation |
| `TIME_LIMIT_HOURS` / `BRANCH_A_RESERVE_HOURS` | — | Kaggle's time limit, and the time held back for Branch A |

**Tuning `OWN_MODEL_WEIGHT`.** Try `0.1` as a separate submission. If the public score goes up, try `0.2`. If it goes down, go back to `0`. Don't tune this weight in validate mode: Branch S was trained on those same 58 studies, so its scores there are inflated.

**Validate mode honesty.** Validate mode doesn't use separate code. It points the submitted pipeline at the labelled studies. Those studies belong to the training data, though, so compare *differences* between versions rather than absolute levels.

**Robustness**
- Branch A must succeed. If it fails, the notebook stops.
- Branches B and S are optional. An error or time-out there is logged, and the run continues.
- During Kaggle's scoring run, anything that can't change the scored file switches itself off.
- Teaching cells run inside `AX_cell`, which restores any variable they overwrite, so analysis code can't corrupt the pipeline.

---

## Outputs

| File | Contents |
|---|---|
| `submission.csv` | One row per study, 12 finding columns, percentile ranks in [0, 1] |
| `final_blend_receipt.json` | Which branches were published, and with which weights |
| `analysis_summary.json` | Label audit, ablation, demo metrics, validation results |
| `diagnostics/branch_b_audit.json` | Every irregularity Branch B tolerated |
| `_sota_0949.csv`, `_branch_b_final.csv`, `_branch_s_final.csv`, … | Intermediate prediction sets, used by Part H |

---

## Results

### Leaderboard

| Version | Public LB |
|---|---|
| First notebook: Branch B, including RadImageNet | 0.943 |
| **This notebook: Branch A (default settings)** | **0.949** |

### LLM pseudo-label quality (Part C)

The labels extracted from reports reach a macro AUC of **0.870** against the official labels (95 % CI 0.831–0.900, n = 57).

### Preprocessing ablation (Part G)

These runs use the ConvNeXt reader on the 58 labelled studies, with one step replaced at inference only:

| Variant | Macro AUC | Δ | Verdict |
|---|---|---|---|
| Reference preprocessing | 0.915 | — | — |
| No millimetre crop | 0.916 | +0.001 | inside the noise |
| Min–max instead of percentile window | 0.916 | +0.001 | inside the noise |
| **Slices sorted by file name** | **0.884** | **−0.031** | **clear loss** |
| No canonical orientation | 0.915 | −0.001 | inside the noise |
| Single slice instead of 3 neighbours | 0.908 | −0.007 | inside the noise |

> 💡 **Slice order is the one step whose removal clearly hurts.** "Inside the noise" means 58 studies are too few to measure the effect, not that the step is useless.

### Live training demo (Part F.3)

This uses frozen ConvNeXt features, 96 training studies and 40 epochs:

| Head | AUC vs LLM labels (held out) | AUC vs official labels |
|---|---|---|
| Mean pooling + linear | 0.842 | 0.813 |
| Attention pooling per finding | **0.866** | **0.823** |

> ⚠️ Every AUC in this README comes from **training** studies. Test labels are hidden, and the leaderboard is the only leakage-free measurement.

---

## Lessons learned

**🐛 A plotting cell silently disabled a model.**
A visualisation cell rebound `studies` from ID strings to `Path` objects. A5 then found no metadata for any study, marked them all "not applicable", and fell back to the previous blend. There was no error, and the submission still looked valid.
**Fix:** namespace isolation (`AX_cell`), plus an input-contract gate before A5 and a coverage gate after it.

**🐛 `strict=False` loaded zero weights.**
Our RadImageNet ResNet-50 reported `missing keys: 267 | unexpected: 318`, which means none of its weights were loaded. The heads trained on it were fitted to a random network that can't be rebuilt, so they're excluded.
**Lesson:** load weights strictly, or count what was actually loaded.

**🐛 A wrong data path skipped analyses quietly.**
The data moved to `/kaggle/input/competitions/…`, and three analysis sections printed "skipped" instead of failing. Path resolution now tries both locations and reports what it found.

**🔒 Integrity checks fail loudly.**
SHA-256 hashes on checkpoints, stored output fingerprints for the 20 DINOv2 members, and strict state-dict loading all stop the run instead of letting the predictions change silently.

---

## Known limitations

- The 0.949 score is for the default settings. Any blend weight above 0 is untested.
- A1 predicts on a black volume when a study can't be decoded, and doesn't record it.
- For sagittal series, A2's flip test-time augmentation swaps front and back, which isn't an anatomical mirror. The scored behaviour was kept.
- Fusion weights in Branches A and B were inherited unchanged, and how they were chosen isn't documented.
- The four B5 readers run from external scripts, which are hash-checked but not shown.
- Offline, A2 logs a harmless `pip` error for `pylibjpeg-libjpeg`.
- Checkpoints load with `torch.load(weights_only=False)`. Only attach datasets you trust.

## Next steps

- [ ] Leaderboard-test `OWN_MODEL_WEIGHT` at 0.1, then 0.2
- [ ] Retrain the RadImageNet heads with correct, strict weight loading
- [ ] Fine-tune the DINOv2 backbone of Branch S instead of freezing it
- [ ] Use A1's anatomical mirror in A2, and compare in validate mode
- [ ] Train the attention head on all 4,349 report-labelled studies, and fit blend weights on out-of-fold predictions

---

## Data handling

- Report text is processed in memory only and never written to outputs. The examples in Part B are invented.
- No manual or hand-tuned edits are made to the predictions.

## Credits

All inputs are public on Kaggle. Thanks to their authors:

| Component | Author |
|---|---|
| CoAtNet weights | `nartaa` |
| ConvNeXt 2.5D reader (code and weights) | `goodpjw2008` |
| Fusion cell (*RSNA Knee Apex Grandmaster Stack V5*) | credits `matterhorn3838` and Jiwei Liu |
| DINOv2 | Meta (Apache-2.0); knee weights by `pilkwang` |
| A5 weights | `mattiaangeli` |
| Further Branch B weights | `dreaddevelopment`, `antoinegg1`, `prvsiyan` |
| RadImageNet ResNet-50 file | via `marwanmath` |
| LLM report labels | `pilkwang` |
| **Branch S model, training, and this notebook** | **`harshikasipani7`** |
