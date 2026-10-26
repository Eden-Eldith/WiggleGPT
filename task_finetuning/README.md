# WiggleGPT: supervised task adaptation, 26 October 2026

Independent full-model adaptations of the original 600,000-step WiggleGPT checkpoint reached **90.1% on SST-2 validation** and **59.5% on RACE-middle test** after three complete epochs per task.

| Task and scored split | Before task training | After task training | Majority baseline |
|---|---:|---:|---:|
| SST-2 sentiment, validation | 437/872 (50.1%) | **786/872 (90.1%)** | 444/872 (50.9%) |
| RACE-middle reading, test | 385/1,436 (26.8%) | **854/1,436 (59.5%)** | 388/1,436 (27.0%) |

“Before” is the original pretrained backbone with the same newly initialized task classifier and task-marker embeddings, evaluated before the first task-training update. These are classifier decisions, rather than free-text answers. The two adaptations start independently from pretraining; neither starts from the SmolTalk instruction checkpoint. Every backbone parameter trains, including the oscillator parameters.

The [eight-page demonstration PDF](WiggleGPT_Demonstration.pdf) presents the results and examples. [results/comparison.json](results/comparison.json) contains all 2,308 paired held-out predictions, labels, source-text hashes, confusion matrices, validation histories and checkpoint hashes. It retains only the short excerpts selected for the demonstration; the complete benchmark corpus is obtained from its original sources.

## Training and data

| Setting | SST-2 | RACE-middle |
|---|---:|---:|
| Training examples | 67,349 | 25,421 |
| Validation examples | 872 | 1,436 |
| Scored held-out examples | 872 validation | 1,436 test |
| Complete epochs | 3 | 3 |
| Optimizer updates | 6,315 | 2,385 |
| Validation-selected best update | 6,315 | 2,385 |
| Effective batch | 32 reviews | 32 questions |
| Microbatch / accumulation | 16 / 2 | 2 questions / 16 |
| Peak allocated GPU memory | 3.846 GiB | 6.383 GiB |

Both runs stopped at `epochs_complete` with zero non-finite updates. The loss is task cross-entropy plus 0.5 times auxiliary language-model cross-entropy, averaged within each candidate sequence and then across candidates. Incorrect answer candidates also receive the auxiliary objective. Both use AdamW, peak learning rate 6.25e-5, 0.2% linear warmup followed by linear decay, classifier dropout 0.1, gradient clipping 1.0 and a maximum sequence length of 512. The pretrained backbone's dropout remains 0.0. Oscillator, one-dimensional and bias parameters have zero weight decay.

SST-2 uses `[START] review [CLASSIFY]` and a 768-to-2 classifier. RACE uses `[START] article [DELIMITER] question [DELIMITER] answer [CLASSIFY]` for each of four candidates, scored by one shared 768-to-1 classifier. Candidate feature dropout shares a mask across the four answers. The start, delimiter and classify markers occupy unused padded-vocabulary IDs 50257, 50258 and 50259; original GPT-2 token IDs remain unchanged. Right padding uses 50256. RACE article prefixes are budgeted against the longest answer so every candidate sees the same passage; questions and options remain complete. There were 210 truncated training passages, nine validation passages and 14 test passages; SST-2 had no truncation.

SST-2 validation supplies both checkpoint selection and the reported score because public test labels are withheld. RACE uses validation for selection and its separate test split for the final score. The historical GPT-1 paper reports 91.3% on SST-2 **GLUE test** and 62.9% on RACE-M after **combined middle/high training**. These provide context: WiggleGPT's SST-2 score is validation, and its RACE adaptation uses middle-only training.

## Run from this repository

Use Python 3.11+ and a CUDA-enabled PyTorch build. CPU preparation uses two threads; model construction, training and evaluation require CUDA. Run one GPU job at a time. The saved configuration caps the process at 9 GiB and reserves desktop memory.

```bash
pip install -r task_finetuning/requirements.txt
python task_finetuning/prepare_data.py
python task_finetuning/encode_task_data.py
```

The preparation script downloads the official SST-2 and RACE-middle splits. The bundled GPT-2/r50k merge table is checked against its SHA-256 before encoding. Generated data stays under `task_finetuning/data/` and is excluded from version control. The source manifests and encoding summaries from the completed experiment are under `results/`.

For a new adaptation, place the original pretrained `ckpt.pt` at `checkpoints/pretrained/ckpt.pt` in the repository root, or supply its path explicitly. It must be the original 600,000-step pretraining checkpoint, not a SmolTalk or task checkpoint.

```bash
python task_finetuning/train_tasks.py --task sst2 --starting-checkpoint checkpoints/pretrained/ckpt.pt
python task_finetuning/train_tasks_v3.py --task race_middle --starting-checkpoint checkpoints/pretrained/ckpt.pt
```

These commands run sequentially. The original trainer is retained for SST-2; version 3 is the final RACE trainer. Version 3 crops padding with a CPU epoch plan, projects only valid auxiliary-LM positions and enables native gradient checkpointing only above 384 cropped tokens. The completed RACE run began with version 2 and continued with version 3 from a saved update; source snapshots and provenance are included under `results/`. These changes preserve question order, batch membership and the per-candidate loss objective. A newly started version-3 run has the same recipe but need not be bit-identical to that segmented run.

Runs write `latest.pt`, validation-selected `best.pt`, their configuration, source snapshots and logs under `task_finetuning/runs/`. Resume with the appropriate trainer and a saved `latest.pt`:

```bash
python task_finetuning/train_tasks_v3.py --resume task_finetuning/runs/YOUR_RACE_RUN/latest.pt
```

The saved run configuration controls resume. Keep the original pretraining checkpoint at the path in that configuration. A single interrupt requests a save after the current update; on Windows, leave the process open until the save finishes. If a terminal terminates the process directly, resume from the last completed `latest.pt`.

## Evaluate released task weights

Task checkpoints are separate from the original language-model checkpoint. Download the released inference weights from the [Hugging Face model page](https://huggingface.co/edeneldith/WiggleGPT) and place the files in the paths below, or pass their actual paths. GitHub source archives do not contain the large weights.

```bash
python task_finetuning/evaluate_tasks.py --checkpoint task_adaptations/sst2/model.pt --output task_finetuning/results/sst2_recomputed.json
python task_finetuning/evaluate_tasks.py --checkpoint task_adaptations/race_middle/model.pt --output task_finetuning/results/race_middle_recomputed.json
```

The evaluator accepts both inference-only exports (`wigglegpt_task_inference_v1`) and full training `best.pt` checkpoints. It loads the native backbone and task head strictly, then scores SST-2 validation or RACE-middle test by default. `--split validation` selects validation explicitly. Scoring the hidden-label SST-2 test split is refused. Evaluation needs the prepared token caches and the task weights; the original pretrained checkpoint and optimizer state are unnecessary.

## Files and provenance

The existing root `model_bio.py` and pretraining scripts are unchanged. `native/model_bio.py` is the exact byte-preserved source used for the completed adaptations (SHA-256 `a28201b8194941a75d3ca2e0812e6d2d0eef1b2656cc0c9f8fd0ca65bf96b28b`). The downloaded repository's root model source differs only in comments/docstrings; the dedicated snapshot keeps provenance exact. `task_runtime.py` carries the original loader and exact tokenizer into this portable folder. The distributed training scripts adapt paths and command-line checkpoint selection; original run source snapshots remain in `results/` for reference.

The repository integration was checked for Python syntax, expected file references, source preservation and consistency with the saved prediction counts. No training or evaluation was rerun during packaging. `results/manifest.json` records SHA-256 hashes of this release folder.

Sources: [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), [RACE](https://huggingface.co/datasets/ehovy/race), [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [OpenAI's 2018 announcement](https://openai.com/index/language-unsupervised/), [tiktoken](https://github.com/openai/tiktoken).

P.C. O'Brien (ORCID 0009-0007-3961-1182), Independent Researcher, originated and directed WiggleGPT and this task extension. OpenAI Codex assisted with implementation, analysis and release preparation.
