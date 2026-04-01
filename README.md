# AMRPA + CAM Library

**Adaptive Multi-layer Recursive Preconditioned Attention**  
with **Compressed Attention Memory**

---

## What Is AMRPA?

Standard transformers discard attention patterns after each layer — *attentional amnesia*.  
AMRPA fixes this by storing attention patterns from earlier layers and injecting them into later layers via a gated, decayed memory system.

**4 Claims:**
1. **Smart Gatekeeper** — similarity-based gate G selectively injects memory  
2. **Dynamic Memory Selection** — MLP alpha selects which past layer matters  
3. **Fading Ink** — exponential decay γ^k reduces trust in older patterns  
4. **Adaptive Memory Depth** — log-scaled window per layer depth  

**Results on multi-hop QA:**
| Dataset | Baseline F1 | AMRPA F1 | ΔF1 |
|---|---|---|---|
| HotpotQA | 0.358 | 0.587 | +22.9% |
| 2WikiMultiHop | 0.663 | 0.764 | +10.0% |
| MuSiQue | 0.278 | 0.477 | +19.8% |

---

## Library Structure

```
amrpa_lib/
├── amrpa/
│   ├── config.py          ← AMRPAConfig + CAMConfig
│   ├── model.py           ← AMRPAModel (single entry point)
│   ├── training.py        ← dataset, train_epoch, evaluate
│   ├── core/
│   │   └── amrpa_core.py  ← AMRPA mechanism (4 claims)
│   ├── cam/               ← CAM compression module
│   ├── adapters/
│   │   ├── encoder.py     ← RoBERTa/BERT adapter
│   │   └── decoder.py     ← GPT2 + CAM adapter
│   └── models/
│       └── qa_model.py    ← AMRPAForQA (ready to use)
├── notebooks/
│   ├── test_library_local.ipynb   ← local test
│   └── kaggle_hotpot_train.ipynb  ← Kaggle training
└── test_library.py        ← full test suite
```

---

## Quick Start

### Encoder (span extraction)
```python
from amrpa import AMRPAConfig, AMRPAForQA, PreprocessedQADataset

config = AMRPAConfig.for_encoder(
    d_model=768,
    n_heads=12
)
model = AMRPAForQA(config, model_name='roberta-base')
```

### Any encoder model
```python
from amrpa import AMRPAConfig, AMRPAModel
from transformers import RobertaModel

model = RobertaModel.from_pretrained('roberta-base')
config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
model, state = AMRPAModel.wrap(model, config)

for batch in dataloader:
    state.reset()                    # ← always call before forward
    outputs = model(**batch)
    metrics = state.get_metrics()    # gate_impact, alpha_diversity, etc.
```

### Decoder (generation)
```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained('gpt2')
config = AMRPAConfig.from_hf_config(model.config, arch='decoder')
model, state = AMRPAModel.wrap(model, config)

for sequence in sequences:
    state.reset()                    # ← reset between sequences
    output = model(input_ids)
```

### Auto-detect architecture
```python
config = AMRPAConfig.from_hf_config(model.config, arch='encoder')
```

---

## On Kaggle

```python
# Option 1: Upload amrpa_lib/ as Kaggle dataset
import sys
sys.path.insert(0, '/kaggle/input/your-dataset-name/amrpa_lib')

# Option 2: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/amrpa.git /kaggle/working/amrpa-lib
sys.path.insert(0, '/kaggle/working/amrpa-lib/amrpa_lib')

from amrpa import AMRPAConfig, AMRPAForQA
```

Open `notebooks/kaggle_hotpot_train.ipynb` for the full training pipeline.

---

## Run Tests

```bash
cd amrpa_lib/
python test_library.py
# Expected: 7/7 suites, 68/68 checks
```

---

## Config Reference

```python
AMRPAConfig(
    d_model          = 768,    # model hidden size
    n_heads          = 12,     # attention heads
    arch             = 'encoder',  # 'encoder' | 'decoder' | 'encoder_decoder'
    n_amrpa_layers   = 4,      # layers to apply AMRPA (from end)
    gamma            = 0.9,    # memory decay factor
    epsilon          = 0.001,  # noise for stability
    alpha_temperature= 0.25,   # layer selection temperature
    d_mlp            = 384,    # alpha MLP hidden dim
    dropout          = 0.2,
    diversity_weight = 0.005,  # alpha diversity regularization
    gate_reg_weight  = 0.05,   # gate regularization
    cam = CAMConfig(           # decoder only
        proj_rank    = 16,
        window_size  = 8,
    )
)
```
