"""
Expanded CNN Survey PDF generator using ReportLab.
Adds: architecture diagram, PyTorch code example, classic network comparison table
with Top-1 accuracy / FLOPs, accuracy-vs-model-size scatter plot, benchmark script
reference, and a full reference list (APA format).
"""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ---------- Performance data (ImageNet val, reported values) ----------
PERF_DATA = [
    # name,         top1,  params_M, gflops,  innovation
    ("LeNet-5",     None,  0.06,     0.0003,  "First practical CNN; tanh activations"),
    ("AlexNet",     63.3,  61.0,     0.72,    "ReLU, Dropout, GPU training"),
    ("GoogLeNet",   69.8,  6.8,      1.50,    "Inception module + 1×1 bottleneck"),
    ("VGG-16",      73.4,  138.0,    15.50,   "Deep stacks of 3×3 convolutions"),
    ("ResNet-50",   76.1,  25.0,     4.10,    "Residual (skip) connections"),
    ("MobileNetV2", 72.0,  3.4,      0.30,    "Depthwise separable + inverted residuals"),
    ("EfficientB0", 77.1,  5.3,      0.39,    "Compound depth/width/resolution scaling"),
]


def performance_table():
    header = ["Network", "Year", "Top-1 (%)", "Params (M)", "GFLOPs", "Key Innovation"]
    years  = {"LeNet-5": "1998", "AlexNet": "2012", "GoogLeNet": "2014",
              "VGG-16": "2014", "ResNet-50": "2016",
              "MobileNetV2": "2018", "EfficientB0": "2019"}
    data = [header]
    for name, top1, params, gflops, innov in PERF_DATA:
        top1_s = f"{top1:.1f}" if top1 is not None else "—"
        data.append([name, years[name], top1_s, f"{params:.1f}", f"{gflops:.2f}", innov])

    available = PAGE_W - 2 * MARGIN
    col_widths = [58, 26, 36, 40, 38, None]
    fixed = sum(w for w in col_widths if w is not None)
    col_widths[-1] = available - fixed

    style = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a1a8c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef0f8")]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN",          (1, 0), (4, -1),  "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    return t


def scatter_plot_image(width_pts: float, height_pts: float) -> RLImage:
    """Return a ReportLab Image of accuracy-vs-model-size scatter plot."""
    fig, ax = plt.subplots(figsize=(width_pts / 72, height_pts / 72), dpi=120)

    for name, top1, params, gflops, _ in PERF_DATA:
        if top1 is None:
            continue
        ax.scatter(params, top1, s=gflops * 20 + 30, zorder=3,
                   color="#1a1a8c", alpha=0.75, edgecolors="white", linewidth=0.5)
        ax.annotate(name, (params, top1),
                    textcoords="offset points", xytext=(6, 2), fontsize=7)

    ax.set_xlabel("Parameters (M)", fontsize=8)
    ax.set_ylabel("Top-1 Accuracy (%) on ImageNet", fontsize=8)
    ax.set_title("Accuracy vs. Model Size  (bubble area ∝ GFLOPs)", fontsize=8.5)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_pts, height=height_pts)


def build_story(styles):
    s = styles
    story = []

    # ---------- Header ----------
    story.append(Paragraph("Convolutional Neural Networks: A Survey", s["title"]))
    story.append(Paragraph(
        "Architecture, Code Examples, Performance Benchmarks, and Recent Trends  ·  March 2026",
        s["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a1a8c"), spaceAfter=5))

    # ---------- Abstract ----------
    story.append(Paragraph("Abstract", s["h1"]))
    story.append(Paragraph(
        "Convolutional Neural Networks (CNNs) are the backbone of modern computer vision, "
        "achieving state-of-the-art results on image classification, object detection, "
        "semantic segmentation, and beyond. This survey traces the evolution of CNNs from "
        "LeNet (1998) to EfficientNet (2019), covering: (i) the core Conv–Pool–FC data-flow "
        "with an annotated architecture diagram; (ii) a minimal PyTorch implementation; "
        "(iii) a quantitative benchmark comparing seven landmark architectures by Top-1 "
        "accuracy, parameter count, and FLOPs; (iv) an accuracy–efficiency scatter plot; "
        "and (v) emerging research directions including CNN–Transformer hybrids, lightweight "
        "edge-deployment models, and self-supervised pre-training.",
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

    # ---------- 4. Performance Comparison ----------
    story.append(Paragraph("4. Performance Benchmark: Classic CNN Architectures", s["h1"]))
    story.append(Paragraph(
        "Table 1 compares landmark architectures by ImageNet Top-1 accuracy (single-crop), "
        "number of parameters, and multiply-accumulate operations (GFLOPs at 224×224 input). "
        "The companion <i>benchmark.py</i> script measures per-sample CPU/GPU latency "
        "using PyTorch <tt>torchvision</tt> models.",
        s["body"],
    ))
    story.append(performance_table())
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Table 1. Landmark CNN architectures on ImageNet. Top-1 accuracy from published papers; "
        "FLOPs computed for a single 224×224 image.",
        s["caption"],
    ))

    # ---------- 5. Scatter Plot ----------
    story.append(Paragraph("5. Accuracy vs. Model Size", s["h1"]))
    story.append(Paragraph(
        "Figure 2 visualises the accuracy–efficiency trade-off. "
        "Bubble area is proportional to GFLOPs. "
        "MobileNetV2 and EfficientNet-B0 achieve competitive accuracy "
        "at a fraction of VGG-16's parameter count.",
        s["body"],
    ))
    plot_w = PAGE_W - 2 * MARGIN
    plot_h = plot_w * 0.48
    story.append(scatter_plot_image(plot_w, plot_h))
    story.append(Paragraph(
        "Figure 2. Top-1 ImageNet accuracy vs. number of parameters (bubble area ∝ GFLOPs). "
        "LeNet-5 omitted (no ImageNet baseline).",
        s["caption"],
    ))

    # ---------- 6. Recent Trends ----------
    story.append(Paragraph("6. Recent Trends", s["h1"]))
    story.append(Paragraph(
        "(a) <b>Neural Architecture Search (NAS)</b> automates topology design for given hardware "
        "budgets (EfficientNet [7]). "
        "(b) <b>Attention mechanisms</b> augment CNNs with channel/spatial attention (CBAM, "
        "Non-local Networks). "
        "(c) <b>Vision Transformers (ViT)</b> [8] replace convolutions with patch-based "
        "self-attention, competitive on large-scale data. "
        "(d) <b>Efficient inference</b> via quantisation, pruning, and knowledge distillation "
        "targets edge deployment.",
        s["body"],
    ))

    # ---------- 7. Conclusion ----------
    story.append(Paragraph("7. Conclusion", s["h1"]))
    story.append(Paragraph(
        "CNNs have transformed computer vision over three decades, evolving from LeNet's "
        "handwritten-digit recogniser to architectures achieving superhuman performance on "
        "ImageNet. Key contributions—shared-weight convolutions, residual connections, and "
        "compound scaling—have steadily improved the accuracy–efficiency frontier. "
        "Looking ahead, three directions are most promising: "
        "(1) <b>CNN–Transformer hybrids</b> (e.g., ConvNeXt, EfficientViT) that blend "
        "local inductive biases with global attention for improved generalisation; "
        "(2) <b>Lightweight and hardware-aware networks</b> combining NAS, quantisation, and "
        "pruning for real-time inference on edge devices; and "
        "(3) <b>Self-supervised and multimodal pre-training</b> (CLIP, MAE) that leverages "
        "unlabelled data at scale, reducing reliance on annotated datasets. "
        "CNNs thus remain—whether standalone or as backbone components—central to AI progress.",
        s["body"],
    ))

    # ---------- References (APA format) ----------
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.grey, spaceBefore=5, spaceAfter=3))
    story.append(Paragraph("References", s["h1"]))
    refs = [
        "[1] Krizhevsky, A., Sutskever, I., &amp; Hinton, G. E. (2012). ImageNet classification with "
        "deep convolutional neural networks. <i>Advances in Neural Information Processing Systems</i>, "
        "<i>25</i>, 1097–1105.",

        "[2] LeCun, Y., Bottou, L., Bengio, Y., &amp; Haffner, P. (1998). Gradient-based learning "
        "applied to document recognition. <i>Proceedings of the IEEE</i>, <i>86</i>(11), 2278–2324. "
        "https://doi.org/10.1109/5.726791",

        "[3] Simonyan, K., &amp; Zisserman, A. (2015). Very deep convolutional networks for "
        "large-scale image recognition. In <i>Proceedings of the International Conference on "
        "Learning Representations (ICLR)</i>. https://arxiv.org/abs/1409.1556",

        "[4] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image "
        "recognition. In <i>Proceedings of the IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR)</i> (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90",

        "[5] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., "
        "Vanhoucke, V., &amp; Rabinovich, A. (2015). Going deeper with convolutions. In "
        "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition "
        "(CVPR)</i> (pp. 1–9). https://doi.org/10.1109/CVPR.2015.7298594",

        "[6] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., &amp; Chen, L.-C. (2018). "
        "MobileNetV2: Inverted residuals and linear bottlenecks. In <i>Proceedings of the IEEE "
        "Conference on Computer Vision and Pattern Recognition (CVPR)</i> (pp. 4510–4520). "
        "https://doi.org/10.1109/CVPR.2018.00474",

        "[7] Tan, M., &amp; Le, Q. V. (2019). EfficientNet: Rethinking model scaling for "
        "convolutional neural networks. In <i>Proceedings of the 36th International Conference "
        "on Machine Learning (ICML)</i> (pp. 6105–6114). https://arxiv.org/abs/1905.11946",

        "[8] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, "
        "T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., &amp; Houlsby, N. "
        "(2021). An image is worth 16×16 words: Transformers for image recognition at scale. In "
        "<i>Proceedings of the International Conference on Learning Representations (ICLR)</i>. "
        "https://arxiv.org/abs/2010.11929",
    ]
    for r in refs:
        story.append(Paragraph(r, s["ref"]))

    return story


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
