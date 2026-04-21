# Clinical NLP — Named Entity Recognition & Summarisation

A small clinical NLP system built from scratch in PyTorch, designed to process 
clinical transcription notes and perform two core tasks:

- **Named Entity Recognition (NER)** — extracting clinical entities such as 
  symptoms and diagnoses from unstructured clinical text, mapped to SNOMED-CT 
  classification
- **Summarisation** — condensing clinical notes into structured 
  impression/plan summaries using weak supervision derived from native 
  document structure

This project is built entirely from first principles — custom tokeniser, 
transformer architecture, and training pipeline — as a deliberate learning 
exercise in understanding how clinical NLP systems work under the hood.

---

## Motivation

Most clinical AI work sits at the API and integration layer. This project exists 
to build genuine understanding of the underlying architectures: how attention 
mechanisms process clinical tokens, how entity span detection works, and how 
summarisation models learn from weakly supervised signal. That understanding 
directly informs better evaluation and governance of third-party clinical AI 
tools in real healthcare settings.

---

## Architecture

### Phase 1 — Clinical NER
A BERT-style encoder-only transformer with a token classification head. The 
model reads clinical notes bidirectionally and assigns BIO tags to each token, 
identifying symptom and diagnosis spans.

### Phase 2 — Clinical Summarisation
A T5-style encoder-decoder seq2seq model trained to condense full clinical 
transcriptions into structured summaries, using ASSESSMENT and PLAN sections 
as weak supervision targets.

### Phase 3 (Planned) — Integrated Pipeline
NER output feeds structured entity context into the summarisation model, 
creating a hybrid pipeline where extracted diagnoses and symptoms inform the 
generated summary.

### Phase 4 (Planned) — Agentic Extension
The integrated pipeline becomes an agent with clinical tools — drug interaction 
lookup, ICD-10 coding, clinical pathway routing.

---

## Data

Training data is sourced from 
**[MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)** 
— a publicly available dataset of ~5000 de-identified medical transcription 
samples across 40 clinical specialties.

Annotation for NER training uses a **weak supervision bootstrap approach**:
- Keywords column provides candidate medical terms per transcription
- **scispaCy** classifies terms as symptoms or diagnoses
- Section structure (ASSESSMENT, SUBJECTIVE, PLAN etc.) provides a 
  confidence signal for entity type
- High confidence labels only are used for training

Evaluation uses the **n2c2 2010 challenge dataset** — a separately sourced, 
manually annotated clinical NER dataset, ensuring clean separation between 
training signal and evaluation.

---

## Status

See [PROGRESS.md](./PROGRESS.md) for current status. 

---

## Dependencies

- Python 3.11
- PyTorch 2.2
- scispaCy
- HuggingFace Datasets
- scikit-learn
