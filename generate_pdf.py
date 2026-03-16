"""
Expanded CNN Survey PDF generator using ReportLab.
Adds: architecture diagram, PyTorch code example, classic network comparison
table (Top-1 accuracy, parameters, FLOPs), accuracy-vs-size scatter plot,
key innovation summaries, and a full IEEE-formatted reference list.
"""
import io

import matplotlib
matplotlib.use("Agg")   # headless backend – no display required
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "SurveyTitle",
        parent=styles["Normal"],
        fontSize=15,
        leading=19,
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=13,
        spaceBefore=7,
        spaceAfter=3,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a8c"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    code = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontSize=7,
        leading=10,
        fontName="Courier",
        backColor=colors.HexColor("#f5f5f5"),
        leftIndent=6,
        rightIndent=6,
        spaceBefore=3,
        spaceAfter=3,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        spaceAfter=5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )
    ref = ParagraphStyle(
        "Ref",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=11,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2,
    )
    return dict(title=title, subtitle=subtitle, h1=h1, body=body,
                code=code, caption=caption, ref=ref)


# ---------------------------------------------------------------------------
# Diagram & code snippets
# ---------------------------------------------------------------------------

ASCII_DIAGRAM = """\
  Input Image                        Output
  (H x W x C)                       Scores
       |                               ^
  +----v---------+               +-----+-----+
  | Conv2d+ReLU  |               | FC (cls)  |
  | (k=3, s=1)   |               | Softmax   |
  +--------------+               +-----------+
       |                               ^
  +----v---------+               +-----+-----+
  | Conv2d+ReLU  |               | FC + ReLU |
  | (k=3, s=1)   |               +-----------+
  +--------------+                     ^
       |                         +-----+-----+
  +----v---------+               |  Flatten  |
  | MaxPool2d    |               +-----------+
  | (k=2, s=2)   |                     ^
  +--------------+    ...repeat...     |
       |_____________________________|"""

PYTORCH_CODE = """\
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    \"\"\"Conv2d -> ReLU -> MaxPool2d -> FC pipeline (CIFAR-10 style).\"\"\"
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B,32,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (B,32,16,16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (B,64,8,8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 64*8*8)
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),  # (B, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

# Quick smoke-test
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)   # batch=4, RGB 32x32
logits = model(x)                 # shape: (4, 10)
print(logits.shape)               # -> torch.Size([4, 10])"""


# ---------------------------------------------------------------------------
# Table 1 – ImageNet performance comparison
# ---------------------------------------------------------------------------

# Columns: Architecture | Year | Top-1 Acc. | Params | GFLOPs | Key Innovation
IMAGENET_DATA = [
    ["Architecture",  "Year", "Top-1 Acc.", "Params",  "GFLOPs", "Key Innovation (one line)"],
    ["LeNet-5",       "1998", "—",          "0.06 M",  "< 0.001", "First practical CNN; gradient-based end-to-end training"],
    ["AlexNet",       "2012", "56.5 %",     "61 M",    "0.72",   "ReLU activations, Dropout, multi-GPU training"],
    ["VGG-16",        "2014", "71.3 %",     "138 M",   "15.5",   "Uniform 3×3 convolutions show depth beats large filters"],
    ["GoogLeNet",     "2014", "69.8 %",     "6.8 M",   "1.5",    "Inception module with parallel 1×1/3×3/5×5 convolutions"],
    ["ResNet-50",     "2016", "76.1 %",     "25 M",    "4.1",    "Residual skip connections enable very deep networks"],
    ["MobileNetV2",   "2018", "71.8 %",     "3.4 M",   "0.30",   "Inverted residuals + depthwise separable convolutions"],
    ["EfficientNet-B0","2019","77.1 %",     "5.3 M",   "0.39",   "Compound scaling of depth, width, and resolution"],
]


def imagenet_table():
    available = PAGE_W - 2 * MARGIN
    # Network | Year | Top-1 | Params | GFLOPs | Key Innovation
    col_widths = [70, 26, 44, 34, 36, None]
    fixed = sum(w for w in col_widths if w is not None)
    col_widths[-1] = available - fixed

    style = TableStyle([
        ("BACKGROUND",      (0, 0), (-1, 0),  colors.HexColor("#1a1a8c")),
        ("TEXTCOLOR",       (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",        (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",        (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS",  (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef0f8")]),
        ("GRID",            (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN",           (1, 0), (4, -1),  "CENTER"),
        ("VALIGN",          (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",      (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 3),
        ("LEFTPADDING",     (0, 0), (-1, -1), 4),
    ])
    t = Table(IMAGENET_DATA, colWidths=col_widths)
    t.setStyle(style)
    return t


# ---------------------------------------------------------------------------
# Figure 2 – Accuracy vs. Model Size scatter plot
# ---------------------------------------------------------------------------

# (name, params_M, top1_acc)  – LeNet-5 excluded (no ImageNet number)
SCATTER_DATA = [
    ("AlexNet",        61.0,  56.5),
    ("VGG-16",        138.0,  71.3),
    ("GoogLeNet",       6.8,  69.8),
    ("ResNet-50",      25.0,  76.1),
    ("MobileNetV2",     3.4,  71.8),
    ("EfficientNet-B0", 5.3,  77.1),
]

# Color palette (one per model)
_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def scatter_image(width_pts: float, height_pts: float) -> RLImage:
    """Generate accuracy-vs-size scatter plot and return a ReportLab Image."""
    dpi = 150
    fig_w = width_pts / 72     # points → inches
    fig_h = height_pts / 72

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    names   = [d[0] for d in SCATTER_DATA]
    params  = np.array([d[1] for d in SCATTER_DATA])
    acc     = np.array([d[2] for d in SCATTER_DATA])

    for i, (name, p, a) in enumerate(zip(names, params, acc)):
        ax.scatter(p, a, s=90, color=_COLORS[i], zorder=3, label=name)
        # nudge label slightly to avoid overlap
        offset_x = 2 if p < 100 else -6
        offset_y = 0.3
        ax.annotate(
            name,
            xy=(p, a),
            xytext=(p + offset_x, a + offset_y),
            fontsize=7,
            color=_COLORS[i],
        )

    ax.set_xlabel("Parameters (M)", fontsize=8)
    ax.set_ylabel("ImageNet Top-1 Accuracy (%)", fontsize=8)
    ax.set_title("Accuracy vs. Model Size", fontsize=9, fontweight="bold")
    ax.set_xlim(-5, 155)
    ax.set_ylim(50, 82)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

    # Efficiency trend arrow
    ax.annotate(
        "More efficient →",
        xy=(10, 75), fontsize=7, color="#555555",
        style="italic",
    )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = RLImage(buf, width=width_pts, height=height_pts)
    return img


# ---------------------------------------------------------------------------
# Story builder
# ---------------------------------------------------------------------------

def build_story(styles):
    s = styles
    story = []

    # ---------- Header ----------
    story.append(Paragraph("Convolutional Neural Networks: A Survey", s["title"]))
    story.append(Paragraph(
        "Architecture, Code Examples, ImageNet Benchmarks, and Recent Trends  ·  March 2026",
        s["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a1a8c"), spaceAfter=5))

    # ---------- Abstract ----------
    story.append(Paragraph("Abstract", s["h1"]))
    story.append(Paragraph(
        "Convolutional Neural Networks (CNNs) are the backbone of modern computer vision, "
        "achieving state-of-the-art results on image classification, detection, segmentation, "
        "and beyond. This survey covers the core CNN data-flow (illustrated with a diagram), "
        "provides a minimal PyTorch implementation, compares landmark architectures on the "
        "ImageNet benchmark (Top-1 accuracy, parameter count, and FLOPs), visualises the "
        "accuracy–efficiency trade-off, and discusses current research directions.",
        s["body"],
    ))

    # ---------- 1. Introduction ----------
    story.append(Paragraph("1. Introduction", s["h1"]))
    story.append(Paragraph(
        "Inspired by the mammalian visual cortex, CNNs exploit local spatial correlations through "
        "shared-weight filters, dramatically reducing parameter counts compared to fully-connected "
        "networks while achieving translation equivariance. Since AlexNet [1] won the 2012 ImageNet "
        "challenge by a large margin, CNNs have become foundational in computer vision, medical "
        "imaging, autonomous driving, and natural-language processing via text-as-image "
        "representations.",
        s["body"],
    ))

    # ---------- 2. Architecture ----------
    story.append(Paragraph("2. Core Architecture and Data-Flow Diagram", s["h1"]))
    story.append(Paragraph(
        "A CNN alternates <b>convolutional</b> and <b>pooling</b> layers, followed by "
        "<b>fully-connected (FC)</b> layers. Convolutions apply learnable filters to local "
        "receptive fields; max-pooling halves spatial resolution at each stage; FC layers perform "
        "final classification. Batch Normalisation and Dropout are standard regularisers.",
        s["body"],
    ))
    story.append(Preformatted(ASCII_DIAGRAM, s["code"]))
    story.append(Paragraph(
        "Figure 1. Schematic data-flow of a typical CNN. Stacked Conv+ReLU blocks are "
        "downsampled by MaxPool; spatial features are flattened and fed to FC classification layers.",
        s["caption"],
    ))

    # ---------- 3. PyTorch Code ----------
    story.append(Paragraph("3. Minimal PyTorch Implementation", s["h1"]))
    story.append(Paragraph(
        "The snippet below shows the Conv2d → ReLU → MaxPool2d → FC pipeline "
        "targeting CIFAR-10 (32×32 RGB, 10 classes):",
        s["body"],
    ))
    story.append(Preformatted(PYTORCH_CODE, s["code"]))
    story.append(Paragraph(
        "The <i>features</i> module handles spatial extraction; "
        "<i>classifier</i> maps flattened features to class logits. "
        "Training uses cross-entropy loss with the Adam optimiser.",
        s["body"],
    ))

    # ---------- 4. ImageNet Performance Table ----------
    story.append(Paragraph("4. ImageNet Performance Comparison", s["h1"]))
    story.append(Paragraph(
        "Table 1 compares landmark CNN architectures on ILSVRC ImageNet, reporting "
        "Top-1 single-crop accuracy, parameter count, and multiply–add operations (GFLOPs) "
        "for a single 224×224 image. GFLOPs is a hardware-independent proxy for computational "
        "cost. Key innovations driving each generation are summarised in the final column.",
        s["body"],
    ))
    story.append(imagenet_table())
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Table 1. Landmark CNN architectures on ImageNet ILSVRC. "
        "Top-1 accuracy: single-crop, 224×224. "
        "GFLOPs: multiply–add operations per image. "
        "LeNet-5 was designed for 32×32 digit images; no standard ImageNet result exists.",
        s["caption"],
    ))

    # ---------- 5. Accuracy vs. Model Size ----------
    story.append(Paragraph("5. Accuracy vs. Model Size", s["h1"]))
    story.append(Paragraph(
        "Figure 2 plots ImageNet Top-1 accuracy against parameter count for the architectures "
        "in Table 1 (LeNet-5 excluded). The scatter illustrates the efficiency frontier: "
        "EfficientNet-B0 and MobileNetV2 achieve competitive accuracy with dramatically fewer "
        "parameters than VGG-16 or AlexNet, demonstrating the impact of architectural innovations "
        "such as depthwise separable convolutions and compound scaling.",
        s["body"],
    ))

    avail_w = PAGE_W - 2 * MARGIN
    plot_h  = avail_w * 0.52          # ~52% aspect ratio
    story.append(scatter_image(avail_w, plot_h))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Figure 2. ImageNet Top-1 accuracy vs. number of parameters (M) for classic CNN "
        "architectures. Models toward the upper-left corner offer the best accuracy–efficiency "
        "trade-off.",
        s["caption"],
    ))

    # ---------- 6. Recent Trends ----------
    story.append(Paragraph("6. Recent Trends", s["h1"]))
    story.append(Paragraph(
        "(a) <b>Neural Architecture Search (NAS)</b> automates topology design for given hardware "
        "budgets (EfficientNet [6]). "
        "(b) <b>Attention mechanisms</b> augment CNNs with channel/spatial attention (CBAM, "
        "Non-local Networks). "
        "(c) <b>Vision Transformers (ViT)</b> [7] replace convolutions with patch-based "
        "self-attention, competitive on large-scale data. "
        "(d) <b>Efficient inference</b> via quantisation, pruning, and knowledge distillation "
        "targets edge deployment.",
        s["body"],
    ))

    # ---------- 7. Conclusion ----------
    story.append(Paragraph("7. Conclusion", s["h1"]))
    story.append(Paragraph(
        "CNNs have transformed computer vision, evolving from LeNet's digit recogniser to "
        "architectures achieving superhuman performance. Hybrid CNN–Transformer models and "
        "self-supervised pre-training continue to push accuracy–efficiency frontiers, ensuring "
        "CNNs remain central to AI progress.",
        s["body"],
    ))

    # ---------- References (IEEE format) ----------
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.grey, spaceBefore=5, spaceAfter=3))
    story.append(Paragraph("References", s["h1"]))
    refs = [
        '[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep '
        'convolutional neural networks," in <i>Advances in Neural Information Processing Systems '
        '(NeurIPS)</i>, vol. 25, 2012, pp. 1097\u20131105.',

        '[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied '
        'to document recognition," <i>Proceedings of the IEEE</i>, vol. 86, no. 11, '
        'pp. 2278\u20132324, Nov. 1998.',

        '[3] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale '
        'image recognition," in <i>International Conference on Learning Representations '
        '(ICLR)</i>, 2015. [Online]. Available: https://arxiv.org/abs/1409.1556',

        '[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image '
        'recognition," in <i>Proc. IEEE/CVF Conference on Computer Vision and Pattern '
        'Recognition (CVPR)</i>, 2016, pp. 770\u2013778.',

        '[5] C. Szegedy et al., "Going deeper with convolutions," in <i>Proc. IEEE/CVF CVPR</i>, '
        '2015, pp. 1\u20139.',

        '[6] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional '
        'neural networks," in <i>Proc. International Conference on Machine Learning (ICML)</i>, '
        '2019, pp. 6105\u20136114.',

        '[7] A. Dosovitskiy et al., "An image is worth 16\u00d716 words: Transformers for image '
        'recognition at scale," in <i>International Conference on Learning Representations '
        '(ICLR)</i>, 2021. [Online]. Available: https://arxiv.org/abs/2010.11929',

        '[8] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: '
        'Inverted residuals and linear bottlenecks," in <i>Proc. IEEE/CVF CVPR</i>, 2018, '
        'pp. 4510\u20134520.',
    ]
    for r in refs:
        story.append(Paragraph(r, s["ref"]))

    return story


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_pdf(filename: str = "cnn_survey.pdf") -> None:
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="CNN Survey",
        author="Survey Generator",
    )
    story = build_story(build_styles())
    doc.build(story)
    print(f"PDF written to {filename}")


if __name__ == "__main__":
    build_pdf("cnn_survey.pdf")
