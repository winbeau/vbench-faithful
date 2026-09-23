# Visual Genome / SNLI annotation audit

Audit date: 2026-09-18. This is an **annotation inventory, not training-set generation or training approval**. Only the two sources below were inspected. No images, videos, weights, teacher calls, model execution, or datasets-server rows were used. Raw downloads and machine-readable manifests/reports live under ignored `data/raw/visual_genome/` and `data/raw/snli/`; do not add these files to Git.

## Sources, licenses and immutable provenance

- Visual Genome: [author-hosted relationships archive](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip). The [author HF script pinned at `65bc9e7e7353fff750326c9523e384701934e530`](https://huggingface.co/datasets/ranjaykrishna/visual_genome/blob/65bc9e7e7353fff750326c9523e384701934e530/visual_genome.py) identifies relationships latest version **1.4.0**, this author-host URL, and **CC BY 4.0** dataset metadata. Its Python code itself has an Apache 2.0 header; that is not the annotation license. The script is downloaded as provenance only, never executed (its builder could fetch images). Builder configs still list 1.0/1.2 and TODO 1.4 support; this audit directly parses the latest raw archive, not an HF builder configuration. The author-host archive is mutable, not revision-addressed: the observed hash below pins our actual bytes; the HF revision pins the URL/version evidence, not future archive contents. HTTP Last-Modified was `Wed, 28 Jun 2023 18:40:02 GMT`.
- SNLI: [official Stanford HF card pinned at `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`](https://huggingface.co/datasets/stanfordnlp/snli/blob/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/README.md), `plain_text/{train,validation,test}-00000-of-00001.parquet`, **CC BY-SA 4.0**. Keep attribution and share-alike obligations when distributing derived annotations; do not assume SNLI and VG licensing are identical. See also the [SNLI project](https://nlp.stanford.edu/projects/snli/).
- **Do not visit `visualgenome.org`: it is now unrelated casino content.** This downloader never requests that domain and rejects redirects to it (the broader source audit had already identified its unrelated content). Old URLs embedded in source metadata are not followed.

All SHA-256 values below were computed on downloaded files, not inferred from ETags:

| Source/file | Actual bytes | SHA-256 |
| --- | ---: | --- |
| VG `visual_genome.py` metadata | 18,057 | `c031e26a97bdb4b97633c091975e879ecbbf43abe71c91c6f4e43be9612c9da9` |
| VG `relationships.json.zip` | 77,904,473 | `e648867b8087e4aeb10019a959f914e2580909f1c3e50a21b25de8741679182f` |
| SNLI `README.md` metadata | 15,970 | `f5182390b149b02e045286a586b7f4d4d626e1e47eade1bc1feabc01ee549578` |
| SNLI train parquet | 19,614,612 | `ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620` |
| SNLI validation parquet | 413,157 | `00f5ed8deaed007fef3022f0215b287efbab815b1bb31ac3f3ff4f4129d41ffe` |
| SNLI test parquet | 411,531 | `4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c` |

VG source total **77,922,530 bytes**, SNLI total **20,455,270 bytes**, each below the 150,000,000-byte compressed/transfer budget including metadata. The VG JSON zip member is 743,673,397 bytes uncompressed; it was streamed one image record at a time, not extracted or loaded whole. Region descriptions were **not downloaded**. Both inventories cover full selected annotation files, not bounded samples. `manifest.json` records URL, retrieval UTC, size, SHA-256, ETag/Last-Modified where supplied.

## Visual Genome: raw relations versus usable supervision

Full actual scan: **108,077 image records / 108,077 distinct image IDs; 2,316,104 relationship records**. Predicate matching lowercases and collapses whitespace, then applies an exact whitelist (no substring labeling):

| Direction | Accepted raw strings | Raw matching records | Structurally valid unique candidates |
| --- | --- | ---: | ---: |
| left | `left of`, `to the left of`, `on the left of` | 390 | 376 |
| right | `right of`, `to the right of`, `on the right of` | 229 | 220 |
| above | `above` | 13,844 | 12,707 |
| below | `below`, `under`, `underneath`, `beneath` | 24,695 | 21,659 |
| **Total** | | **39,158** | **34,962** |

All observed left/right whitelist hits are `left of`/`right of`. Below-family raw counts: `below` 3,371; `under` 18,459; `underneath` 1,509; `beneath` 1,356. Of 39,158 matched records, 121 fail nonempty endpoint name / endpoint ID / non-self checks and 4,075 repeat `(image_id, subject.object_id, canonical_direction, object.object_id)`. The other **2,276,946** raw relationships are outside this whitelist. Alias object IDs/merged IDs are not resolved, so remaining candidates are not necessarily semantically distinct.

For comparison only, raw whole-word predicate mentions are left **2,162**, right **1,819**, above **15,442**, below **3,607**. Those broad lexical counts include unsupported phrasings and are **not labels**. Generic `on` occurs **667,990** times and is excluded: `on` does not generally mean above. `over`, `next to`, front/behind, etc. are not mapped by this audit.

Observed structure (abridged real relationship, not a natural prompt):

```json
{"image_id":3,"relationship":{"relationship_id":15974,"predicate":"under","subject":{"object_id":1060247,"names":["computer tower"],"merged_object_ids":[5096],"x":216,"y":351,"w":79,"h":129},"object":{"object_id":5097,"name":"desktop","x":129,"y":233,"w":510,"h":244}}}
```

Endpoints mix singular `name` with plural `names`; relation records also carry synsets and object geometry. Direction is subject relative to object. The raw IDs are retained in inspection examples. The first six accepted unique examples are preserved in ignored `inspection.json`, not fabricated natural captions.

**Real usable natural prompt/label pairs approved here: 0.** 34,962 is a pool of structurally eligible graph edges only, not 34,962 training samples or independently authored prompts. `under`/`beneath` may describe support, cover, or contextual relations rather than the desired image-plane geometry; even exact `above` may be contextual/noisy. No geometry truth check, region-text alignment, entity disambiguation, human quality review, or template generation was performed. If later authorized, generated subject–predicate–object sentences must be labeled **synthetic weak supervision**, not human natural descriptions. Do not assign every relationship in an image as the target for a local region description; targets must be supported by that particular text. The severe left/right scarcity (596 total structurally unique candidates) is an explicit coverage limitation.

## SNLI: auxiliary NLI, not Scene ground truth

All three pinned parquet files are downloaded separately. Local Python has no pyarrow; the optional inspection uses the existing remote project's `.venv` with pyarrow 25.0.1, without changing dependencies or using GPUs. Its file hashes are checked against the local downloaded bytes. Statistics below refer to actual batch-scanned rows, not just the dataset card.

| Split | Raw rows | Unlabeled rejected | Valid labeled rows | Exact labeled duplicates | Unique labeled rows | Conflicting text pairs | Unambiguous unique pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 550,152 | 785 | 549,367 | 547 | 548,820 | 64 | 548,677 |
| validation | 10,000 | 158 | 9,842 | 2 | 9,840 | 0 | 9,840 |
| test | 10,000 | 176 | 9,824 | 0 | 9,824 | 0 | 9,824 |

No valid-labeled row had an empty premise/hypothesis after stripping surrounding whitespace. Dedup key is `(stripped_premise, stripped_hypothesis, label)` within each split; no case/punctuation normalization. A conflicting text pair has multiple distinct valid labels: removing all 143 distinct labeled rows belonging to the 64 conflicting train pairs leaves **548,677** unambiguous unique auxiliary NLI candidates, **not approved Scene samples**. Dev/test stay reserved, not added to that training candidate count.

Actual raw label histograms (entailment / neutral / contradiction / unlabeled): train **183,416 / 182,764 / 183,187 / 785**; validation **3,329 / 3,235 / 3,278 / 158**; test **3,368 / 3,219 / 3,237 / 176**. The card's nominal 10,000 dev/test rows therefore must not be mistaken for 10,000 valid labels.

Cross-split exact text-pair intersections, regardless of label among valid-labeled rows: **train–validation 1; train–test 0; validation–test 0**. Quarantine the overlap against held-out evaluation before any future training; the table is a per-split inventory before cross-split purging, not a finalized leakage-free corpus. Broader caption/image-family overlap remains unmeasured.

Observed schema: `premise: string`, `hypothesis: string`, `label: int64`. Real first train records share the premise `A person on a horse jumps over a broken down airplane.`; hypotheses are `A person is training his horse for a competition.` (1), `A person is at a diner, ordering an omelette.` (2), and `A person is outdoors, on a horse.` (0). Inspection retains three examples per split. The full reports are local `data/raw/snli/inspection.json` and remote `~/wenbiao_zhao/vbench-prompts-compile/data/raw/snli/inspection.json`; original split files remain separate in both locations. Remote script copy is ignored `data/raw/snli/inspect_vg_snli.py`, not a remote tracked-code change.

Label mapping from the pinned metadata: **0 entailment, 1 neutral, 2 contradiction; -1 unlabeled**. Correct NLI direction is premise entails hypothesis; a prose field in the card reverses that direction, so do not copy that wording into a mapping. For any later auxiliary Scene mapping, evidence caption corresponds to the premise and requested prompt claim to the hypothesis, not vice versa. Entailment/contradiction/neutral could become supported/contradicted/insufficient **only as a reviewed auxiliary text-NLI task**, not as measured real-pipeline labels.

**Real usable prompt + Tag2Text caption Scene training pairs approved here: 0.** SNLI premises are human captions, not Tag2Text outputs; labels are textual judgments, not scene-specialized visual evidence labels. `neutral` is insufficient evidence, not proof of visual absence. People/action biases and hypothesis-only annotation artifacts remain. Preserve official splits; never merge validation/test into training. Text-exact dedup is not image/source-family dedup. The parquet schema lacks original image/pair identifiers, so row index + pinned split/file hash is the auditable local source key, not a recovered image key. Flickr30k and Visual Genome contribute source captions to SNLI: cross-dataset leakage cannot be ruled out by different dataset names or a zero exact pair overlap. Before joint training/evaluation recover source IDs where possible, conservatively group caption/image/derived families, and keep independent human gold held out.

## Reproduce and verify

From the project root, with Python 3.11.14:

```bash
# Network opt-in; hard 150 MB per-source cap, annotation-only fixed allowlisted files.
.venv/bin/python scripts/inspect_vg_snli.py --download --download-only
# Offline VG reinspection:
.venv/bin/python scripts/inspect_vg_snli.py --source visual_genome
# On an already provisioned pyarrow environment, keeping splits separate:
.venv/bin/python scripts/inspect_vg_snli.py --source snli
```

Without optional pyarrow, SNLI inspection raises an explicit error directing the user to the existing remote environment and preserves any previous report rather than inventing counts. Offline CLI inspection verifies file size, hash, and pinned source URL from the manifest before analysis. The downloader validates cached files against its prior manifest hash and refuses unknown existing files; it never invokes upstream dataset loaders. The saved report is inventory only; no training JSONL is emitted.

Validation: CLI help; synthetic empty/two-item/chunk-boundary/truncated JSON-array smoke checks; synthetic predicate filtering with duplicate removal and exclusion of generic `on`; name/names normalization; full VG streaming scan. SNLI validation details are recorded with actual counts above. No GPU/model/teacher/training validation was attempted.
