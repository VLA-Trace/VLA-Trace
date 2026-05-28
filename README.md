# 🔍 VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing

## 🧭 Overview

> This repository will host the official implementation of **VLA-Trace**, a diagnostic framework for understanding how Vision-Language-Action models convert multimodal knowledge into embodied control.

**VLA-Trace** studies VLA models as evolving, controllable systems rather than opaque end-to-end policies. It builds a progressive evidence chain from **representation dynamics**, to **causal control attribution**, to **closed-loop behavioral manifestation**.

<p align="center">
  <img src="assets/overview.png" alt="VLA-Trace overview" width="100%"/>
</p>

Modern VLA models inherit powerful vision-language priors, but policy learning can reshape those priors in subtle ways. VLA-Trace asks where multimodal knowledge is preserved, which pathways are actually used for action decoding, and when visually grounded behavior still fails to follow fine-grained semantic changes.

## 🗞️ News

- **[Coming Soon]** 📄 Paper link will be released.
- **[Coming Soon]** 💻 Code, configs, and reproduction scripts will be released.
- **[Coming Soon]** 🌐 Project page, visualizations, and result artifacts will be released.

## ✨ Highlights

- 🧬 **Representation-level diagnosis:** trace how visual, textual, and joint representations evolve from pretrained VLMs to pretrained and finetuned VLAs.
- 🔌 **Causal control attribution:** use attention knockout to test which modality routes are necessary for action decoding.
- 🧪 **Behavior-level validation:** connect internal pathways to rollout attention, visual shortcut dependence, and semantic editing behavior.
- 📊 **Architecture-aware findings:** compare pi-style and OpenVLA-style policies to reveal different adaptation and routing patterns.
- 🛠️ **Release-oriented toolkit:** planned utilities include CKA extraction, knockout interventions, rollout localization, patch masking, input editing, and plotting scripts.

## 🧩 Diagnostic Pipeline

| Stage                            | Core Question                                | Main Probe                                  | What It Reveals                                                        |
| -------------------------------- | -------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| 🧬**Representation Trace** | What changes during VLA adaptation?          | Cross-modal CKA + checkpoint-drift CKA      | Preservation or reorganization of visual, textual, and joint subspaces |
| 🔌**Causal Pathway Trace** | Which modalities control action decoding?    | Prefill and generation attention knockout   | Causal visual/text/action routes and layer-wise bottlenecks            |
| 🧪**Behavior Trace**       | How do internal pathways appear in rollouts? | Attention IoU, patch masking, input editing | Grounding, shortcut dependence, and semantic controllability           |

## 🔬 Method

### 🧬 Stage 1: Representation Shifts under VLA Adaptation

We compare internal representations across three model stages:

| Symbol | Stage          | Description                                          |
| ------ | -------------- | ---------------------------------------------------- |
| `C0` | Pretrained VLM | General vision-language prior before action learning |
| `C1` | Pretrained VLA | VLA policy after action-oriented pretraining         |
| `C2` | Finetuned VLA  | Task-finetuned policy with task-specific weights     |

VLA-Trace uses:

- **Cross-modal CKA** to measure layer-wise image-text alignment.
- **Checkpoint-drift CKA** to measure whether vision, text, and joint subspaces are preserved or reorganized across `C0`, `C1`, and `C2`.

### 🔌 Stage 2: Causal Pathways for Action Decoding

To test whether multimodal information is merely represented or actually used, VLA-Trace performs attention knockout interventions:

- **Prefill knockout** blocks vision-language interaction during context formation.
- **Generation knockout** blocks action-token access to image or instruction tokens during decoding.
- **Layer-wise knockout** localizes whether action-critical information flows through narrow bottlenecks or distributed regions.

This turns representation analysis into a causal question: *which pathway must remain open for the robot to act successfully?*

### 🧪 Stage 3: Behavioral Probes of Grounding and Shortcut Dependence

VLA-Trace then moves from internal mechanisms to closed-loop behavior:

- **Attention IoU:** checks whether action attention overlaps with objects, robot regions, and robot-object interaction regions.
- **Temporal attention analysis:** tests whether attention shifts with multi-step task progress.
- **Visual patch masking:** removes target objects, grippers, robot bodies, or backgrounds to expose shortcut reliance.
- **Input editing:** changes objects or instructions to test fine-grained semantic controllability.

## 📊 Key Findings

- 🧬 **Different VLAs adapt different modalities.** pi0.5 shows more fluctuating cross-modal fusion and reorganizes textual representations into task-conditioned control features, while OpenVLA more strongly preserves text-pooled representations and mainly restructures visual/joint subspaces.
- 🔌 **Action decoding follows architecture-specific routes.** pi0.5 is dominated by a concentrated visual-to-action pathway, whereas OpenVLA relies on both visual grounding and prompt-region access.
- 🧪 **Visual grounding does not guarantee semantic flexibility.** Models often track task-relevant objects and robot-object interaction regions, yet may remain biased toward the original visual-task configuration after semantic edits.
- 📉 **Shortcut dependence is model-specific.** Target-object masking is highly destructive across models, while gripper, robot-body, and background masking reveal different reliance patterns across architectures.

## 🤖 Models and Benchmarks

The full diagnostic chain focuses on **pi0.5** and **OpenVLA**, with supplementary experiments on **OpenVLA-OFT** and **X-VLA**.

| Model       | Backbone   | Decoding       | Observation                     | Coverage                             |
| ----------- | ---------- | -------------- | ------------------------------- | ------------------------------------ |
| pi0.5       | PaliGemma  | Flow matching  | RGB + language                  | LIBERO, CALVIN                       |
| OpenVLA     | Prismatic  | Autoregressive | RGB + language                  | LIBERO, CALVIN, Simpler              |
| OpenVLA-OFT | Prismatic  | L1 regression  | RGB + language + proprioception | LIBERO, CALVIN, RoboTwin2.0          |
| X-VLA       | Florence-2 | Flow matching  | RGB + language + soft prompt    | LIBERO, CALVIN, Simpler, RoboTwin2.0 |

Benchmarks include:

- **LIBERO-10:** long-horizon compositional manipulation.
- **LIBERO-Object:** object-centric understanding.
- **LIBERO-Spatial:** spatial reasoning.
- **LIBERO-Goal:** goal-conditioned control.
- **CALVIN, SimplerEnv-BridgeV2, and RoboTwin2.0:** broader validation across sequential control, real-to-sim transfer, and bimanual manipulation settings.

## 🛠️ Repository Status

Code is currently being organized for public release.

### 🚀 Coming Soon

- [ ] 🧬 CKA extraction and representation visualization
- [ ] 📐 Checkpoint-drift analysis utilities
- [ ] 🔌 Attention knockout implementation
- [ ] 🎯 Rollout attention localization tools
- [ ] 🧱 Visual patch masking suite
- [ ] ✏️ Input editing protocols
- [ ] ⚙️ Benchmark and model configuration files
- [ ] 📊 Reproducible plotting scripts

## 🗂️ Planned Structure

```text
VLA-Trace/
├── assets/                  # Figures and README media
├── configs/                 # Model, benchmark, and probe configs
├── vla_trace/
│   ├── representations/     # Cross-modal and checkpoint-drift CKA
│   ├── knockout/            # Attention knockout interventions
│   ├── rollouts/            # Closed-loop rollout utilities
│   ├── perturbations/       # Visual patch masking and input editing
│   └── visualization/       # Plotting and qualitative analysis
├── scripts/                 # Reproduction entry points
├── examples/                # Minimal runnable examples
└── README.md
```

## 🔗 Links

The following links are placeholders and will be updated upon release:

| Resource                 | Link        |
| ------------------------ | ----------- |
| 📄 Paper                 | Coming soon |
| 🌐 Project Page          | Coming soon |
| 💻 Code                  | Coming soon |
| 📦 Artifacts             | Coming soon |
| 📊 Results / Leaderboard | Coming soon |

## 📄 Paper

The paper PDF will be linked here after release. For now, a local draft PDF is included in this folder:

- [`VLA_Trace__EMNLP2026_.pdf`](./VLA_Trace__EMNLP2026_.pdf)

## 📚 Citation

If you find **VLA-Trace** useful, please consider citing our work:

```bibtex
@misc{vlatrace2026,
  title  = {VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 📬 Contact

For questions, collaborations, or release updates, please open an issue or contact the authors after the repository is public.

## 🙏 Acknowledgements

VLA-Trace builds on the growing ecosystem of open VLA models, robotic manipulation benchmarks, and mechanistic interpretability tools. We thank the developers of pi-style models, OpenVLA, LIBERO, CALVIN, SimplerEnv, and RoboTwin for enabling reproducible research in embodied AI.

## 📜 License

The license will be announced with the public code release.
