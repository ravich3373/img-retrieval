# Model survey — text→image retrieval with adjectives & composed objects

> **Goal:** pick models to test in this harness for the question *"when objects carry qualifying
> adjectives (red car) and several objects are composed in one query (a red car left of a wooden
> chair), what retrieves the right image?"*
>
> **Verification status (read this):** First drafted 2026-06-08 from model training knowledge,
> then the **open-weight first-wave specs were verified live on the web (June 2026)** and corrected.
> Verified runnable: `google/siglip2-so400m-patch16-384` (SigLIP 2, arXiv 2502.14786, Feb 2025);
> `openclip:ViT-H-14-378-quickgelu:dfn5b` (Apple DFN5B); OpenCLIP tags `laion2b_s34b_b79k` (ViT-B-32)
> and `laion2b_s39b_b160k` (ViT-bigG-14); `openclip:EVA02-L-14:merged2b_s4b_b131k`;
> `jinaai/jina-clip-v2` (transformers, trust_remote_code, `.encode_text`/`.encode_image`; needs
> einops+timm); `facebook/metaclip-2-worldwide-huge-quickgelu` (MetaCLIP 2, Aug 2025); NegCLIP from
> the ARO repo (ViT-B/32, use the **quickgelu** arch). **Still from memory / reconfirm:** exact
> benchmark numbers (ImageNet %, COCO R@1), and the entire **hosted-API** section (pricing/dims) —
> the latter is out of scope for now (we're testing open-weight models first). Confidence: H/M/L below.

---

## TL;DR — what to wire up first

**The headline finding that should shape the whole project:** *every* model below is a contrastive
**dual-encoder**, and that entire family is structurally weak at **attribute binding** and **word
order**. On ARO / SugarCrepe / Winoground most CLIP variants score near chance on the binding tests.
Scaling data (bigG, DFN5B, SigLIP 2) buys only modest gains. Real improvements come from (a)
hard-negative fine-tuning (NegCLIP), (b) long text context (Jina), or (c) structure/syntax
supervision. **So this harness should bracket the phenomenon**: include a known-weak baseline
(OpenAI CLIP), a known-improved one (NegCLIP), and the current quality frontier (SigLIP 2 / DFN5B).

### Open weights — first wave
1. **SigLIP 2** — `google/siglip2-so400m-patch16-384` (`transformers`). Current best open dual-encoder tier; Apache-2.0; NaFlex sibling handles non-square images. *Default backbone.*
2. **DFN5B** — `apple/DFN5B-CLIP-ViT-H-14-378` (`open_clip`). Top classic-CLIP retrieval/zero-shot. (Check Apple license.)
3. **OpenAI CLIP L/14-336** — `openai/clip-vit-large-patch14-336` (`transformers`). The reference *failure baseline*; MIT.
4. **NegCLIP** — checkpoint from the ARO repo (`open_clip`). The *compositionally-improved* comparison point.
5. **Jina-CLIP v2** — `jinaai/jina-clip-v2` (`transformers`, `trust_remote_code`). Long text (8192 tok) + multilingual; only one that ingests long multi-attribute queries without truncation.
6. **EVA-02-CLIP-L/14** — `openclip:EVA02-L-14:merged2b_s4b_b131k` (`open_clip`/`timm`) or **OpenCLIP ViT-bigG/14** — `openclip:ViT-bigG-14:laion2b_s39b_b160k` — extra fully-open (MIT) high-quality data point + architectural diversity.

### Hosted APIs — DEFERRED (not testing API-only models for now; kept for reference)
1. **Voyage `voyage-multimodal-3`** — shared 1024-d space, simple SDK, strong retrieval.
2. **Cohere `Embed v4`** — long context, Matryoshka dims + int8/binary, clean REST.
3. **Jina `jina-embeddings-v4`** (or `jina-clip-v2`) — cheapest; multi-vector mode useful for compositional probing.
4. **TwelveLabs `Marengo`** — *video-native* (text/image → clip); the one to add if/when the corpus becomes video (directly relevant to a video-AI product).
5. **Google Vertex `multimodalembedding@001`** — if already on GCP; mature, video support, low-dim options.

> OpenAI `text-embedding-3-*` and chat VLMs (GPT-4o, Gemini) are **excluded from cross-modal
> retrieval**: text-embedding-3 is text-only, and GPT-4o/Gemini are generators, not shared-space
> embedders. A VLM is only usable here as a *reranker/judge*, not a retriever.

---

## 1. Open-weight CLIP-family dual encoders

| Model / variants | HF id(s) + loader | Embed dim / res | Retrieval strength | Compositional / binding notes | License | Conf |
|---|---|---|---|---|---|---|
| **OpenAI CLIP** (B/32, B/16, L/14, L/14-336) | `openai/clip-vit-base-patch32`, `…-large-patch14-336` — `transformers`/`open_clip` | 512 (B), 768 (L); 224/336 | Baseline. COCO T→I R@1 ~30–37% | The original bag-of-words failure case; ~chance on ARO order/relation. **Keep as negative baseline.** | MIT | H |
| **OpenCLIP (LAION)** — `ViT-H/14`, `g/14`, `bigG/14`, DataComp-L | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `laion/CLIP-ViT-L-14-DataComp.XL-…` — `open_clip` | H/g 1024, bigG 1280; 224 | Strong; bigG/G top-tier classic CLIP | Same weakness; cleaner data → marginal ARO gains | MIT (code) | H |
| **SigLIP** — base/large/so400m @224/256/384 | `google/siglip-so400m-patch14-384`, `google/siglip-base-patch16-224` — `transformers`(Siglip)/`open_clip` | base 768, so400m 1152 | Strong; so400m-384 SOTA-class for size | Sigmoid loss doesn't fix binding; CLIP-like profile | Apache-2.0 | H |
| **SigLIP 2** (2025) — base/large/so400m + **NaFlex** + giant | `google/siglip2-so400m-patch16-384`, `…-naflex`, `google/siglip2-giant-opt-patch16-384` — `transformers`(Siglip2)/`open_clip` | so400m 1152, giant ~1536; 224–512 + NaFlex | **Current top open tier.** Beats CLIP/SigLIP on zero-shot+retrieval | Dense/self-distill objectives → better localization & *some* binding gains; still contrastive | Apache-2.0 | H (IDs M) |
| **EVA-CLIP / EVA-02** — L/14, bigE, 8B, 18B | open_clip `EVA02-L-14:merged2b_s4b_b131k` (timm `eva02_large_patch14_clip_224.merged2b_s4b_b131k`), `QuanSun/EVA-CLIP` — `open_clip`/`timm` | L 768, larger else; 224/336 | Very strong per-FLOP; 18B top open zero-shot at release | Standard CLIP binding weakness | MIT | H |
| **MetaCLIP** v1 — B/16, L/14, H/14 | `facebook/metaclip-h14-fullcc2.5b` — `transformers`/`open_clip` | B 512, L/H 768/1024; 224 | Matches/beats OpenAI CLIP via curation | CLIP-class | CC-BY-NC (verify) | H (lic M) |
| **MetaCLIP 2** "worldwide" (Aug 2025) — multilingual | `facebook/metaclip-2-worldwide-huge-quickgelu` (+ `-giant`, `-s16-384`) — `transformers` (MetaClip2Model)/`open_clip` | ~1024 (H); 224/378 | Competitive w/ SigLIP 2 tier; multilingual | Same dual-encoder limits | CC-BY-NC (verify) | H (id) |
| **DFN** (Apple Data Filtering Nets) — DFN2B, DFN5B | `apple/DFN5B-CLIP-ViT-H-14-378`, `apple/DFN2B-CLIP-ViT-L-14` — `open_clip` | L 768, H 1024; 224/378 | Excellent per-FLOP; DFN5B-H/14-378 ~84% IN | Data filtering → robustness, not binding | Apple ASCL (verify commercial) | H (lic M) |
| **AIMv2** (Apple, autoregressive) — incl. `-lit`/aligned | `apple/aimv2-large-patch14-224`, `…-lit` — `transformers`/`timm` | L 1024+; 224–448 | Strong backbone; use the **aligned/-lit** variant for text→image | Not a binding fix | Apple ASCL | M |
| **Jina-CLIP v2** (2024) | `jinaai/jina-clip-v2` — `transformers` (trust_remote_code) | 1024 (Matryoshka); 512px | Strong multilingual + **long text (8192 tok)** | Long context helps detailed multi-attribute queries vs CLIP's 77-tok cap | CC-BY-NC (commercial via Jina) | H |
| **NLLB-CLIP** (200+ langs) | `visheratin/nllb-clip-large-siglip` — `open_clip`/`transformers` | ~768/1152; 224/384 | Best language coverage; English below SigLIP2/DFN | Inherits CLIP/SigLIP limits | CC-BY-NC (verify) | M |

**Compositionality-focused fine-tunes**
| Model | Access | Binding note | Conf |
|---|---|---|---|
| **NegCLIP** (Yuksekgonul et al., ARO) | `open_clip`-loadable ckpt from `vision-language-models-are-bows` repo | Canonical hard-negative fine-tune (swapped attrs/relations) → large ARO gains over base CLIP. **Essential "improved" comparison.** | M |
| **Structure/syntax-aware CLIP** (SVLC, Structure-CLIP, syntax-aware) | mostly research repos | Scene-graph / syntax supervision → ARO/SugarCrepe/Winoground gains, often trading raw retrieval. Availability spotty. | L |

**Vision-only backbones** (no text tower → not directly usable for text→image without a text head):
**DINOv2** (`facebook/dinov2-large`, Apache-2.0) and **DINOv3** (2025) — top image↔image retrieval; relevant only if you add a LiT-style text aligner.

---

## 2. Hosted multimodal embedding APIs

> **Deferred:** we're not testing API-only models for now. This table is kept for reference;
> figures are from memory (Jan 2026) and unverified. The `voyage:`/`cohere:`/`jina:` backends exist
> in `models/hosted.py` but are not registered by default — re-enable when needed.

| Provider / model | Image+text? | Shared space? | Dims | API | Pricing ⚠️ | Notes | Conf |
|---|---|---|---|---|---|---|---|
| **Cohere Embed v4** | yes (+PDF/screenshot pages) | yes | 1536 default; Matryoshka 256/512/1024/1536 | `/v2/embed`, `input_type` search_document/query | ~$0.12/1M tok ⚠️ | Long context; int8/binary out; strong on doc/screenshot | M |
| **Cohere Embed v3 (multimodal)** | yes | yes | 1024 | `/v1/embed` `input_type=image` | ~$0.10/1M ⚠️ | Superseded by v4; 1 image/req | M |
| **Voyage `voyage-multimodal-3`** | yes (interleaved) | yes | 1024 | `/v1/multimodalembeddings`, `voyageai` SDK | ~$0.12/1M ⚠️ | Top-tier mixed text+image; screenshots/figures | M |
| **Jina `jina-embeddings-v4`** | yes (single- or multi-vector) | yes | 2048 single (Matryoshka→128); multi-vec 128/tok | Jina `/v1/embeddings`; HF weights | ~$0.02–0.05/1M ⚠️ | 3.8B; late-interaction strong for fine-grained/compositional | M |
| **Jina `jina-clip-v2`** | yes | yes | 1024 (Matryoshka→64) | Jina API | ~$0.02/1M ⚠️ | CLIP-style; cheaper/smaller than v4; 89 langs | H |
| **Google Vertex `multimodalembedding@001`** | yes (+**video**) | yes | 1408 (also 128/256/512) | Vertex `predict`, GCP SDK | per-image + per-sec-video + per-1K-char ⚠️ | Mature, GCP-native, video; low-dim options for cheap ANN | M |
| **Amazon Titan Multimodal G1** | yes | yes | 1024 (also 384/256) | Bedrock `InvokeModel` | ~$0.06/1M + ~$0.00006/img ⚠️ | AWS-native baseline | M |
| **Nomic Embed Multimodal (3B/7B) / Vision v1.5** | yes | yes (Vision v1.5 aligned to Embed Text v1.5) | Vision 768; Multimodal multi-vector (ColPali-style) | Nomic Atlas API; open weights | Atlas low / self-host free ⚠️ | Open+hosted; strong on ViDoRe visual-doc | M |
| **Marqo** (ecommerce/GCL) | yes | yes | ~768–1024 | Marqo Cloud index API | index/compute-based ⚠️ | E-commerce product image↔text; tied to Marqo vector DB | M |
| **TwelveLabs `Marengo` (2.6/2.7)** | **video**+image+text+audio | yes (video-native) | ~1024 ⚠️ | TwelveLabs `/embed` (+async video) | per-minute video ⚠️ | **Video-native — most relevant if corpus is video.** text/image→clip | M |
| **OpenAI `text-embedding-3`** | **text-only** | n/a | 1536/3072 | `/v1/embeddings` | $0.02/$0.13 per 1M | **Cannot embed images — excluded.** | H |
| **GPT-4o / Gemini** | VLM (chat), **not embedder** | n/a | — | chat | — | Usable only as a *reranker/judge*. | H |

---

## 3. Composed Image Retrieval (CIR) — image **+** text query

> Use these when the query becomes *reference image + modifier text → target image* (e.g. "this car
> but red"). For pure text→image attribute binding you don't need CIR — you need a strong CLIP/SigLIP
> scorer plus the right *evaluation* (§4).

| Method | Year | Idea | Access | Adjective relevance |
|---|---|---|---|---|
| **MagicLens** | ICML'24 (Google) | Self-supervised single/dual-encoder on ~36.7M⚠️ (img, instruction, img) web triplets w/ LLM instructions; SOTA CIR at small size | `github.com/google-deepmind/magiclens` (JAX) + community PyTorch ⚠️ | **Very high** — broadest instruction coverage incl. attribute/relation edits. Best image+text query encoder. |
| **CIReVL** | ICLR'24 | **Training-free**: VLM captions ref image → LLM edits caption → text retrieval | `github.com/ExplainableML/Vision_by_Language` ⚠️ | **High** — edit happens in language, so "red→blue" is explicit/interpretable |
| **Pic2Word** | CVPR'23 (Google) | ZS-CIR: map image→single pseudo-word token, compose in text prompt | `github.com/google-research/composed_image_retrieval` ⚠️ | Moderate — single token loses fine attributes |
| **SEARLE** | ICCV'23 | ZS-CIR textual inversion + GPT-phrase regularization; introduced **CIRCO** | `github.com/miccunifi/SEARLE` ⚠️ | Moderate–good |
| **LinCIR** | CVPR'24 | Language-only training w/ random keyword replacement (itself an attribute-swap aug) | `github.com/navervision/lincir` ⚠️ | Moderate–high |
| **CompoDiff** | TMLR'24 (NAVER) | Latent diffusion denoises target CLIP embedding from ref img+text; supports **negative text**; trained on SynthTriplets18M | `github.com/navervision/CompoDiff` ⚠️ | **High** — negative conditioning + the synthetic-triplet recipe is a template for our generator |

Also: MLLM-reranker CIR (LLaVA/Qwen-VL/GPT-4V), Context-I2W/KEDs/FTI4CIR (incremental ZS-CIR), CoVR/CoVR-2 (composed **video** retrieval, if we extend to video).

---

## 4. Benchmarks & the evaluation protocol we mirror

### Compositionality / attribute-binding probes (image–text *matching* w/ hard negatives)
| Benchmark | Tests | Hard-negative construction | Metric | Code ⚠️ |
|---|---|---|---|---|
| **SugarCrepe** (NeurIPS'23) | REPLACE/SWAP/ADD × {object, attribute, relation} | ChatGPT-generated **then adversarially filtered** so a blind LLM can't cheat | image→text acc per category | `RAIVNLab/sugar-crepe` |
| **ARO** (ICLR'23) | Attribution, Relation, Order | swap two attributes between objects; swap nouns around a relation; permute word order | text→image acc per subset | `mertyg/vision-language-models-are-bows` |
| **Winoground** (CVPR'22) | same words, different order | hand-curated minimal pairs | **text / image / group score** | `facebook/winoground` (gated, 400 items) |
| **CREPE** (CVPR'23) | systematicity & productivity | atom swaps / compositional perturbations from VG scene graphs | Recall@K / HN-Recall | `RAIVNLab/CREPE` |
| **VL-CheckList** | Objects/Attributes/Relations taxonomy (color, material, size, state…) | one-word negative caption | ITM acc per category | `om-ai-lab/VL-CheckList` |
| **EqBen** (ICCV'23) | equivariance to minimal changes | minimal-change *image* pairs (Kubric/video) | group-score | `Wangt-CN/EqBen` |

### CIR retrieval benchmarks
**CIRR** (R@K + R_subset@K on visually-similar hard set), **CIRCO** (mAP@K, multiple ground-truths on COCO), **FashionIQ** (R@10/50, fashion).

### → Protocol implemented in this harness
Borrowing SugarCrepe's discipline + Winoground's scoring, on our synthetic gallery:

1. **Full-gallery retrieval:** Recall@{1,5,10} and MRR over all images. Overall quality.
2. **Hard-negative recall by type:** restrict gallery to {1 positive + matched distractors that differ by exactly one factor}, report rank-1 accuracy split by negative type:
   - **color/attribute swap** (same object, wrong color) → attribute-value binding
   - **object swap** (same color, wrong object) → object grounding
   - **binding swap** (two-object query, swap *which* object owns the attribute; same words) → the critical compositional binding test
   - **relation swap** (left-of ↔ right-of / reordered) → relational/order
   - **add/remove object** → composition completeness
3. **Winoground-style minimal pairs:** for two images × two captions differing only by a swap, compute **text-score / image-score / group-score** (group = strictest binding signal).
4. **Per-complexity buckets:** 1-object / 2-object / 2-object+relation — show degradation with composition depth (CREPE lesson).
5. **Controls:** easy-negative gallery (random images) alongside hard-negative; the **easy-R@1 − hard-R@1 gap** is the compositionality signal. Adversarially sanity-check that a text-blind ranker can't beat chance on the hard negatives.

**Benchmarks worth mirroring as published comparisons:** SugarCrepe (taxonomy + adversarial filter), ARO (attribute/relation swaps), Winoground (scoring protocol only — data too small).
