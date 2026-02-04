# Custom LLM for Pediatric Occupational Therapy

**Model Training and Inference**

This repository contains code supporting the training and inference of a custom fine-tuned large language model (LLM) for generating draft pediatric occupational therapy (OT) SOAP notes from point-form scratch notes, as described in:

DiMaio et al. (2025). A Language Model for Pediatric Occupational Therapy Documentation: Model Development and Pilot.

The code is provided to support transparency and reproducibility of the methodological pipeline. Due to privacy, institutional, and data governance constraints, clinical data and trained model weights are not included.

**Model Training Overview**

The training pipeline supports:
- Supervised fine-tuning of instruction-tuned LLMs (e.g., Llama-3-8B-Instruct)
- Full fine-tuning or parameter-efficient fine-tuning (e.g., LoRA)

Training data are expected to consist of paired:
- Input: point-form scratch notes
- Output: clinician-authored SOAP notes

No example clinical data are included in this repository.
Users must supply their own appropriately de-identified datasets.

**Model Inference Overview**

The model inference pipeline supports:
- Generation of scratch notes with large instruction tuned LLMs (e.g. Llama-2-70B-Chat) using few-shot examples

**Intended Use**

This codebase is intended for:
- Research replication and inspection of the training and inference pipeline
- Adaptation by other institutions using their own de-identified clinical data
- Educational purposes related to applied clinical NLP

**Privacy and Ethics Notice**

This repository does not include:
- Personal health information (PHI)
- Identifiable clinical notes
- Trained model weights derived from real patient data

**License**

This repository is released under the Apache 2.0 License.
