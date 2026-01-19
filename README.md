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
pip install torch transformers peft accelerate
pip install einops einops_exts
pip install faiss-cpu  # or faiss-gpu
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
# Eval-only mode
accelerate launch train/train_rag_decoder.py \
    --config config/radfm/5_graph_pos.yaml \
    --checkpoint ./results/model_best.pt \
    --eval-only
```

## Model Weights

Model checkpoints are available on Hugging Face:
- [gyuang/cpr-rag-models](https://huggingface.co/gyuang/cpr-rag-models)

Download and place in `models/` directory:
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
@article{cpr-rag2024,
  title={CPR-RAG: Clinical Prior-guided Retrieval-Augmented Generation for CT Report Generation},
  author={},
  year={2024}
}
```

## License

This project is for research purposes.
