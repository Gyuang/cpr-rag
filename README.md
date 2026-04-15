# CPR-RAG: Clinical Prior-guided Retrieval-Augmented Generation for CT Report Generation

A retrieval-augmented generation framework for automated CT (computed tomography) report generation. CPR-RAG combines volumetric vision encoders with LLaMA decoders via LoRA fine-tuning, enhanced by clinical knowledge graph-guided retrieval.

## Architecture

```
CT Volume [B, 1, D, H, W]
    ↓
Vision Encoder (RadFM / M3D / CT2Rep)
    ↓
Perceiver/Pooling → [B, num_tokens, dim]
    ↓
Linear Projector + LayerNorm
    ↓
LLaMA Decoder (+ LoRA) ← Retrieved Reports (RAG)
    ↓
Generated Report
```

## Features

- **Multiple Vision Backbones**: Support for RadFM, M3D-CLIP, and CT2Rep encoders
- **Hierarchical RAG**: Organ-specific retrieval with clinical knowledge graph reranking
- **Efficient Training**: LoRA fine-tuning with gradient checkpointing
- **Precomputed Embeddings**: Skip vision encoder forward pass for faster training

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── config/                 # Training configurations
│   ├── base_rag.yaml
│   └── radfm/             # RadFM-specific configs
├── models/
│   ├── core/              # Visual LLaMA, vision backbones
│   ├── vision/            # RadFM, M3D, CT2Rep implementations
│   ├── classifiers/       # Organ classifier
│   └── vlm_factory.py     # Model building utilities
├── rag/
│   ├── hierarchical_retriever.py   # Organ-specific retrieval
│   ├── clinical_graph_reranker.py  # Knowledge graph reranking
│   └── unified_retriever.py        # Multi-retriever fusion
├── train/
│   ├── train_rag_decoder.py        # Main RAG training script
│   ├── train_organ_classifier.py   # Classifier training
│   ├── dataset_ct.py               # CT dataset loaders
│   └── dataset_rag.py              # RAG-aware dataset
├── scripts/
│   └── eval_ce_from_predictions.py # Clinical Efficacy evaluation
└── utils/
    └── radiology_eval.py           # Evaluation metrics
```

## Usage

### Training

```bash
# RAG-aware training with hierarchical retrieval
python train/train_rag_decoder.py \
    --config config/radfm/5_graph_pos.yaml \
    --save-dir ./results/my_experiment

# With Accelerate for multi-GPU
accelerate launch train/train_rag_decoder.py \
    --config config/radfm/5_graph_pos.yaml
```

### Evaluation

```bash
# Eval-only mode (generates predictions CSV)
accelerate launch train/train_rag_decoder.py \
    --config config/radfm/5_graph_pos.yaml \
    --checkpoint ./results/model_best.pt \
    --eval-only

# Clinical Efficacy (CE) evaluation from predictions
python scripts/eval_ce_from_predictions.py \
    --predictions ./results/test_predictions_eval.csv \
    --output ./results/ce_metrics.yaml
```

## Model Weights

Model checkpoints will be available on Hugging Face (coming soon):
- [gyuang/cpr-rag-models](https://huggingface.co/gyuang/cpr-rag-models)

After downloading, place files in `models/` directory:
```
models/
├── checkpoints/
│   └── radfm_organ_classifier.pt
├── graphs/
│   └── ct_rate_condprob.pt
└── rag_indices/
    └── organ_classifier_index_radfm.pkl
```

## Configuration

Key configuration options in YAML files:

| Option | Description |
|--------|-------------|
| `vision_backbone` | Vision encoder: `radfm`, `m3d`, `ct2rep` |
| `retriever_type` | Retrieval method: `hierarchical` |
| `retrieval_top_k` | Number of retrieved examples |
| `lora_r` | LoRA rank |
| `freeze_llama` | Freeze LLaMA weights (LoRA-only) |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yang2026cprrag,
  title={CPR-RAG: Clinical Prior-Regularized Retrieval for Anatomy-Aware 3D CT Report Generation},
  author={Yang, Sungkyu and Kim, Kang-Min and Kim, Mansu},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year={2026}
}
```

## License
This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
It is intended for academic and research use only. Commercial use is prohibited.
