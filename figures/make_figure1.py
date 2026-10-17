"""Figure 1 - workflow of the XAI-ChestDiag framework (vector, publication quality)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
fig, ax = plt.subplots(figsize=(16, 7.6))
ax.set_xlim(0, 164); ax.set_ylim(0, 76); ax.axis("off")

C = {"in": ("#F2F3F4", "#566573"), "pre": ("#EAF2F8", "#2E86C1"), "fe": ("#E8F8F5", "#17A589"),
     "cls": ("#FEF5E7", "#CA6F1E"), "xai": ("#FDEDEC", "#C0392B"), "out": ("#F4ECF7", "#7D3C98")}

def box(x, y, w, h, key, title, lines):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=11.5, weight="bold", color=ec)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.0 - i * 3.4, t, ha="center", va="top", fontsize=9.5, color="#1B2631")

def arrow(p, q, color="#34495E", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

H = 26
box(1, 25, 24, H, "in", "Input", ["Frontal chest radiograph", "NIH ChestX-ray14", "1024 × 1024 grayscale"])
box(29, 25, 31, H, "pre", "Phase 1: Preprocessing",
    ["Resize to 224 × 224", "Min–max scaling to [0, 1]", "ImageNet standardisation",
     "Training only: rotation ±10°,", "flip, zoom 10%, CLAHE"])
box(65, 25, 34, H, "fe", "Phase 2: Feature Extraction",
    ["DenseNet121 (ImageNet init.)", "4 dense blocks, 3 transition layers",
     "Dense connectivity:", "x_l = H_l([x_0, x_1, …, x_(l−1)])", "Output: 7 × 7 × 1024 maps"])
box(106, 46, 35, H, "cls", "Phase 3: Classification",
    ["Global Average Pooling", "Dropout (p = 0.3)", "Dense layer + Sigmoid",
     "P(abnormal | x)", "Threshold 0.5 → Normal / Abnormal"])
box(106, 3, 35, H, "xai", "Phase 4: Explainability",
    ["Grad-CAM on last dense block", "α_k = mean_ij (∂y / ∂A_k,ij)", "L = ReLU(Σ_k α_k A_k)",
     "Up-sampled to 224 × 224", "Validated by Pointing Game"])
box(146, 25, 17, H, "out", "Output", ["Triage label", "+ probability", "", "Heat-map overlay", "for clinician"])

arrow((25.3, 38), (28.7, 38)); arrow((60.3, 38), (64.7, 38))
arrow((99.3, 44), (105.7, 59), rad=-0.2); arrow((99.3, 32), (105.7, 16), rad=0.2)
ax.text(100, 56, "feature\nmaps", fontsize=8.5, ha="center", color="#555")
ax.text(97.5, 12, "feature maps\n+ gradients", fontsize=8.5, ha="center", color="#555")
arrow((123.5, 45.7), (123.5, 29.3), color="#C0392B")
ax.text(125, 37.5, "class score y\n(back-propagated)", fontsize=8.5, color="#C0392B", va="center")
arrow((141.3, 59), (145.7, 46), rad=-0.2); arrow((141.3, 16), (145.7, 29), rad=0.2)

ax.text(50, 63, "Data: patient-level split 70 / 10 / 20 % (train / validation / test), no patient overlap",
        ha="center", fontsize=10, style="italic", color="#2E4053",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#AAB7B8", lw=1))
for ext in ("png", "pdf"):
    fig.savefig(f"figures/figure1_workflow.{ext}", dpi=300, bbox_inches="tight")
print("saved")
