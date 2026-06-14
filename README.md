# MicroScope AI

Real-time environmental microorganism classification and water-quality screening, built end-to-end: dataset preparation, model training, ONNX export, and a deployed web app.

**Live app:** [microscope-ai-corsmju4n9im8j7sypfwr7.streamlit.app](https://microscope-ai-corsmju4n9im8j7sypfwr7.streamlit.app/)

---

## Overview

MicroScope AI identifies 21 species of environmental microorganisms from microscopy images using a fine-tuned MobileViT-v2 model, and surfaces the result through a Streamlit interface with live analytics and exportable session reports. The goal was to take a small, relatively obscure benchmark dataset (EMDS-6) and turn it into a working, deployable tool — not just a training notebook.

## Result

- **93.65% test accuracy** / **0.935 macro-F1** on EMDS-6 (21 classes, 840 images)
- For context, the previously reported best result on this dataset (Xception, 2022) was **44.29% accuracy** — modern lightweight transformer backbones combined with stronger augmentation and training recipes make a substantial difference on small biomedical image datasets
- Model size: **4.4M parameters**, exported to ONNX for fast CPU inference

## Architecture

```
Training (Google Colab)
  EMDS-6 dataset (21 classes, 840 images)
    -> Albumentations v2 augmentation pipeline
    -> MobileViT-v2 fine-tuning (PyTorch Lightning)
    -> ONNX export
    -> Hosted on Hugging Face Hub

Deployment (Streamlit Community Cloud)
  User uploads image / webcam capture
    -> ONNX Runtime inference (CPU)
    -> Top-3 species prediction + confidence
    -> Risk categorization
    -> Session logged to in-memory DuckDB
    -> Live Plotly dashboard
    -> PDF session report export
```

## Tech stack

| Component | Tool |
|---|---|
| Model | MobileViT-v2 (timm), fine-tuned |
| Training | PyTorch Lightning, Albumentations v2 |
| Inference | ONNX Runtime |
| Model hosting | Hugging Face Hub |
| App framework | Streamlit |
| Analytics | Plotly, DuckDB (in-memory) |
| Reporting | fpdf2 (PDF export), Gemini 1.5/2.0 Flash (optional narrative reports) |

## Features

- **Analyze** — upload an image or use a live webcam capture; get top-3 species predictions with confidence scores and a water-safety risk badge
- **Dashboard** — live session analytics: species frequency, risk distribution, detection timeline, full detection log
- **Report** — AI-generated diagnostic notes (via Gemini, when configured) and a downloadable PDF session summary

## Limitations and future work

This project was built with a deliberate focus on shipping a complete, working pipeline within limited time and free-tier infrastructure. Some tradeoffs are intentional and documented here rather than hidden:

- **Dataset size.** EMDS-6 contains only 40 images per class (840 total). The 93.65% test accuracy is measured on a 126-image held-out test set — strong relative to prior published results, but the dataset's small size means results should be read as a demonstration of method, not a clinically validated benchmark.
- **Segmentation (SAM2) is disabled in the deployed app.** The architecture supports SAM2-based segmentation as a pre-processing step, with automatic fallback to whole-image classification when no GPU is available. Streamlit Community Cloud's free tier is CPU-only, so the app currently runs in classify-full-image mode. Re-enabling SAM2 requires GPU-backed hosting.
- **Risk categorization is illustrative.** The safe / caution / hazardous labels assigned to each species are a simplified heuristic for demonstration purposes and have not been validated against a formal water-quality bioindicator standard. They should not be used for actual water safety decisions without review by a domain expert.
- **Session data is not persistent.** Detection logs are stored in an in-memory DuckDB instance and reset when the app restarts. A production deployment would use a persistent or shared database.
- **Gemini-based reports require an API key** with available quota and are optional — the core classification, dashboard, and PDF export work independently of this feature.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optionally set `GEMINI_API_KEY` in `.streamlit/secrets.toml` to enable AI-generated diagnostic reports.

## Dataset

[EMDS-6 (Environmental Microorganism Dataset, version 6)](https://figshare.com/) — 21 classes of environmental microorganisms, 40 original images and corresponding ground-truth segmentation masks per class.
