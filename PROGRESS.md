# Clinical NLP Project - Progress Tracker

## Data Pipeline
- [x] EDA complete
- [x] `data_cleaning.py` - filter short transcriptions, remove low count specialties, strip boilerplate from keywords, handle NaN keyword recovery from section headers, output to SQLite
- [x] `test_data_cleaning.py` - pytest test suite, 20/20 passing
- [ ] Section header parser - flexible regex for SOAP and non-SOAP templates
- [ ] Bootstrap annotator - scispaCy + section context + keyword signal → BIO tagged sequences
- [ ] Output preprocessed data to Parquet for model training

## NER Model
- [ ] Tokeniser - BPE from scratch
- [ ] BERT-style encoder architecture - embeddings, multi-head attention, feed forward blocks, layer norm
- [ ] NER token classification head
- [ ] Training loop - optimiser, LR schedule, gradient clipping
- [ ] Evaluation harness - F1 per entity class against n2c2 annotated data

## Summarisation Model
- [ ] Encoder-decoder seq2seq architecture
- [ ] Training on ASSESSMENT/PLAN weak supervision pairs
- [ ] Evaluation

## Integration
- [ ] NER output feeding into summarisation as structured context
- [ ] Agentic pipeline design
