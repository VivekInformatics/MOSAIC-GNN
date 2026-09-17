# 🧩 MOSAIC-GNN: Multi-modal Orchestrated Spatial AI Clustering

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph--Neural--Networks-3C2179)](https://pyg.org/)
[![DINOv2](https://img.shields.io/badge/Vision_Backbone-Meta_DINOv2-0081FB)](https://github.com/facebookresearch/dinov2)
[![Scanpy & Squidpy](https://img.shields.io/badge/Single--Cell-Scanpy%20%7C%20Squidpy-brightgreen)](https://squidpy.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MOSAIC** — **M**ulti-modal **O**rchestrated **S**patial **A**I **C**lustering
>
> *Small tiles, big picture. Like a mosaic assembles fragments into art, MOSAIC-GNN fuses gene expression, histology morphology, and spatial topology to reveal the hidden architecture of tissue.*

A self-supervised graph neural network framework that integrates **Spatial Gene Expression**, **Multi-Scale Histology Morphology (DINOv2)**, and **Spatial Tissue Topology** via Residual GATv2 with Channel-Wise Gated Fusion and Clean Spatial Contrastive Learning for tissue domain identification in spatial transcriptomics.

Evaluated on the **10x Genomics Visium Mouse Brain Sagittal Anterior** dataset (2,688 spots).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Architectural Innovations](#-key-architectural-innovations)
- [Model Architecture & Formulation](#-model-architecture--formulation)
- [Ablation Study & Benchmark Results](#-ablation-study--benchmark-results)
- [Biological Marker Gene Validation](#-biological-marker-gene-validation)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Output Artifacts](#-output-artifacts)
- [Methodological Notes & Caveats](#-methodological-notes--caveats)
- [Citation & References](#-citation--references)

---

## 🔬 Overview

Spatial transcriptomics (ST) pairs high-throughput gene expression profiling with spatial coordinate metadata and paired H&E histology slides. Identifying functional tissue domains requires synthesizing all three modalities without allowing one to drown out the others or causing feature over-smoothing across graph neighbors.

MOSAIC-GNN solves this with channel-wise gated residual modulation, multi-scale morphological representation learning, residual graph attention encoders, and true-positive-excluded spatial contrastive loss (InfoNCE).

```
   ┌───────────────────────┐   ┌────────────────────────┐   ┌───────────────────────┐
   │ Spatial Transcriptome │   │  H&E Histology Slide   │   │  Spatial Coordinates  │
   │  3,000 HVGs → 50 PCA  │   │ Local (112) + Ctx (224)│   │   6-NN Spatial Graph  │
   └───────────┬───────────┘   └───────────┬────────────┘   └───────────┬───────────┘
               │                           │ (Meta DINOv2)              │
               ▼                           ▼                            │
      ┌─────────────────┐         ┌─────────────────┐                   │
      │ Gene GATv2 Enc  │         │ Morph GATv2 Enc │                   │
      │ (with Residual) │         │ (with Residual) │                   │
      └────────┬────────┘         └────────┬────────┘                   │
               │                           │                            │
               └─────────────┬─────────────┘                            │
                             ▼                                          │
             ┌───────────────────────────────┐                          │
             │   Channel-Wise Gated Fusion   │                          │
             │  Z_fused = LayerNorm(Zg+g⊙Zm) │                          │
             └───────────────┬───────────────┘                          │
                             │                                          │
                             ▼                                          ▼
             ┌──────────────────────────────────────────────────────────────┐
             │       Self-Supervised Objective:                             │
             │       L = L_recon(Gene) + 0.5 · L_InfoNCE(Spatial, τ(t))    │
             │       (Clean Negative Sampling: Non-Neighbors Only)          │
             └───────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    MOSAIC Domain Assignments   │
                             └───────────────────────────────┘
```

---

## 💡 Key Architectural Innovations

1. **Channel-Wise Gated Residual Modulation:**
   Overcomes scalar gate collapse (where scalar gates clamp to ≈0.28, discarding morphology). Each of the 64 latent dimensions independently decides how much morphological context to incorporate.

2. **Residual GATv2 Graph Encoder:**
   2-layer multi-head GATv2 (4 heads) with learnable linear skip projections and LayerNorm — prevents over-smoothing and stabilizes gradient flow across graph hops.

3. **Clean Spatial InfoNCE (True-Positive Exclusion):**
   Constructs an exact adjacency set lookup and strictly samples negatives from non-neighboring nodes. Traditional random negative sampling risks selecting true spatial neighbors as negatives — MOSAIC eliminates this contamination.

4. **Principled Multi-Scale DINOv2 Feature Fusion:**
   Local (112×112) and contextual (224×224) H&E patches encoded with Meta's `dinov2_vits14`, standardized, weighted (0.70/0.30), concatenated, and jointly PCA-projected into 50 dimensions in a single shared coordinate system.

5. **Exact Cosine Temperature & Warmup LR Scheduling:**
   Cosine τ annealing (τ_max=0.25 → τ_min=0.15) with exact endpoint alignment, linear warmup (10 epochs), cosine LR decay, and gradient clipping (max_norm=1.0).

---

## 📐 Model Architecture & Formulation

### 1. Channel-Wise Gated Fusion

$$g = \sigma\left(W_g\,[Z_{\text{gene}} \,\|\, Z_{\text{morph}}] + b_g\right) \in \mathbb{R}^d$$

$$Z_{\text{fused}} = \text{LayerNorm}\left(Z_{\text{gene}} + g \odot \text{GELU}(W_m\,Z_{\text{morph}})\right)$$

- Gate bias initialized at −1.0 (σ(−1.0) ≈ 0.27) so training begins near the gene-expression manifold and gradually activates morphological channels.

### 2. Loss Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}}^{\text{gene}} + 0.5 \cdot \mathcal{L}_{\text{InfoNCE}}^{\text{spatial}}(\tau(t))$$

$$\mathcal{L}_{\text{InfoNCE}}(\tau) = -\frac{1}{|\mathcal{E}|}\sum_{(i, j) \in \mathcal{E}} \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\exp(\text{sim}(z_i, z_j)/\tau) + \sum_{k=1}^{N_{\text{neg}}} \exp(\text{sim}(z_i, z_{k}^{\text{neg}})/\tau)}$$

Where negatives $z_k^{\text{neg}} \notin \mathcal{N}(i)$ (strictly non-neighbor nodes).

### 3. Temperature Schedule (Exact Endpoints)

$$\tau(t) = \tau_{\min} + \frac{1}{2}(\tau_{\max} - \tau_{\min})\left(1 + \cos\left(\frac{\pi(t-1)}{T-1}\right)\right)$$

---

## 📊 Ablation Study & Benchmark Results

All variants evaluated across **5 independent random seeds** (200 epochs each) on the 10x Visium Mouse Brain dataset (15 reference Leiden clusters).

| Variant | Details | ARI (↑) | NMI (↑) | Spot ACC (↑) | Silhouette (↑) | Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V1** | Gene Only (Plain GATv2) | 0.5558 ± 0.0312 | 0.7060 ± 0.0173 | 64.35% ± 2.88% | 0.1859 ± 0.0038 | N/A |
| **V2** | Concat Fusion + DINOv2 | 0.5308 ± 0.0388 | 0.6975 ± 0.0201 | 63.23% ± 2.79% | 0.1715 ± 0.0055 | N/A |
| **V3** | Channel-Gated + DINOv2 | 0.5842 ± 0.0399 | 0.7248 ± 0.0222 | 68.28% ± 3.74% | 0.1637 ± 0.0039 | 0.395 ± 0.017 |
| **V4** 🏆 | **MOSAIC (Channel-Gated + Residual GATv2)** | **0.6131 ± 0.0197** | **0.7415 ± 0.0144** | **70.54% ± 3.21%** | 0.1640 ± 0.0033 | **0.461 ± 0.015** |
| **V5** | Full (+ EdgeDrop + CosTau + LR Sched) | 0.5403 ± 0.0182 | 0.7019 ± 0.0113 | 63.03% ± 2.05% | **0.2060 ± 0.0031** | 0.364 ± 0.012 |

### Key Findings

- **Gating > Concatenation (V3 vs V2):** Channel-wise gating lifts accuracy by +5.05%, proving unconstrained morphology injection adds noise unless dynamically gated.
- **Residual Connections Matter (V4 vs V3):** Adding residual skip projections to GATv2 yields **70.54% Spot Accuracy** and **0.6131 ARI** — the strongest configuration.
- **Active Morphology Engagement:** The learned gate rose from initial bias 0.27 → 0.461, confirming the model actively leverages histology features rather than suppressing them.

---

## 🧬 Biological Marker Gene Validation

Wilcoxon rank-sum differential expression on MOSAIC-predicted spatial domains confirms biologically coherent marker gene signatures:

| Domain | Top 5 Marker Genes |
| :---: | :--- |
| **0** | *Resp18*, *6330403K07Rik*, *Gpx3*, *Ndn*, *Baiap3* |
| **1** | *1110008P14Rik*, *Dkk3*, *Lmo4*, *Cplx1*, *Scn1b* |
| **2** | *Prkcd*, *Pcp4*, *Rora*, *Adarb1*, *Tnnt1* |
| **3** | *Mef2c*, *Stx1a*, *Satb1*, *Dkkl1*, *Cabp1* |
| **4** | *Cbln1*, *Cbln4*, *Ctxn3*, *Tcf7l2*, *Foxp2* |
| **5** | *Hpcal1*, *Lypd1*, *Nr2f2*, *Hap1*, *Ly6h* |
| **6** | *Hpca*, *Cnih2*, *Cabp7*, *Ncdn*, *Psd* |
| **7** | *Ppp1r1b*, *Ttr*, *Penk*, *Adora2a*, *Gpr88* |
| **8** | *Cldn11*, *Mal*, *Cnp*, *Cryab*, *Mobp* |
| **9** | *Lamp5*, *Atp1a1*, *Mef2c*, *Camk2n1*, *Nrgn* |
| **10** | *Calb2*, *Agt*, *Sparc*, *Dbi*, *Nnat* |
| **11** | *3110035E14Rik*, *Tbr1*, *Ttc9b*, *Ipcef1*, *Trbc2* |
| **12** | *Nptxr*, *Mef2c*, *Nov*, *Snca*, *Stx1a* |
| **13** | *Vamp1*, *Hcn2*, *Qdpr*, *Mobp*, *Rnd2* |
| **14** | *Nptxr*, *Olfm1*, *Ttc9b*, *Slc30a3*, *Slc17a7* |

---

## 📁 Repository Structure

```
MOSAIC-GNN/
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
├── trimodal-gnn-improved-run2_Executed.ipynb        # Main executable notebook
├── data/
│   └── anndata/
│       └── visium_hne_adata.h5ad                   # 10x Visium dataset (auto-downloads)
└── output/
    ├── ablation_study_table.csv                    # Aggregated ablation metrics
    ├── ablation_per_seed_raw.csv                   # Per-seed raw metrics
    ├── domain_predictions.csv                      # Spot-level domain assignments
    ├── spatial_gnn_embeddings.csv                  # 64-dim latent embeddings
    ├── top_marker_genes_per_cluster.csv             # Ranked marker genes per domain
    ├── benchmark_results.png                       # 6-panel diagnostic visualization
    └── silhouette_distribution.png                 # Per-cluster silhouette boxplot
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/MOSAIC-GNN.git
cd MOSAIC-GNN
```

### 2. Create Virtual Environment
```bash
conda create -n mosaic-gnn python=3.10 -y
conda activate mosaic-gnn
```

### 3. Install PyTorch & PyTorch Geometric
```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
```

### 4. Install Additional Dependencies
```bash
pip install scanpy squidpy seaborn pandas scikit-learn scipy matplotlib pillow
```

---

## 🚀 Usage

### Run the Notebook
```bash
jupyter notebook trimodal-gnn-improved-run2_Executed.ipynb
```

### Quick Smoke Test vs. Full Benchmark
Adjust the hyperparameter block at the top of the notebook:
```python
# Quick test
N_SEEDS = 1
TOTAL_EPOCHS = 50

# Full benchmark (publication-level)
N_SEEDS = 5
TOTAL_EPOCHS = 200
```

---

## 📈 Output Artifacts

| File | Description |
| :--- | :--- |
| `benchmark_results.png` | 6-panel figure: spatial maps, t-SNE, loss curves, gate/τ/LR trajectories, ablation bar chart |
| `silhouette_distribution.png` | Per-cluster silhouette boxplot showing separation quality |
| `ablation_study_table.csv` | Mean ± std metrics across all variants |
| `ablation_per_seed_raw.csv` | Full per-seed metric breakdown |
| `domain_predictions.csv` | Spot-level predicted domain labels |
| `spatial_gnn_embeddings.csv` | 64-dimensional normalized latent embeddings |
| `top_marker_genes_per_cluster.csv` | Ranked DE marker genes per predicted domain |

---

## ⚠️ Methodological Notes & Caveats

1. **Reference Labels:** Squidpy's Leiden clustering is used as an internal reference for architecture comparison — not expert-annotated anatomical ground truth.
2. **No External Baselines:** This is an internal ablation study. No comparison to published methods (STAGATE, GraphST, conST, SpaGCN, DeepST).
3. **Multi-Scale Weighting:** The 0.70/0.30 local-to-context weighting is an empirical heuristic, not a tuned hyperparameter.
4. **Cluster Count:** KMeans uses k=15 from the reference labels (not automatically discovered).

---

## 📜 Citation

If you find MOSAIC-GNN useful in your research, please consider citing:

```bibtex
@software{mosaic_gnn_2024,
  title   = {MOSAIC-GNN: Multi-modal Orchestrated Spatial AI Clustering
             via Graph Neural Networks},
  author  = {Your Name},
  url     = {https://github.com/YOUR_USERNAME/MOSAIC-GNN},
  year    = {2024}
}
```

### Acknowledgements
- **10x Genomics** — Visium spatial transcriptomics datasets
- **Meta AI Research** — [DINOv2](https://github.com/facebookresearch/dinov2) vision foundation model
- **Squidpy & Scanpy** — spatial and single-cell omics analysis frameworks
