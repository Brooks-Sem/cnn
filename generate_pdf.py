"""
Expanded CNN Survey PDF generator using ReportLab.
Adds: architecture diagram, PyTorch code example, classic network comparison table,
and a full reference list (7 key papers).
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted
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


def classic_table():
    header = ["Network", "Year", "Depth", "Params", "Top-5 Err.", "Key Innovation"]
    data = [
        header,
        ["LeNet-5",    "1998", "7",   "60 K",   "—",      "First practical CNN; tanh activations"],
        ["AlexNet",    "2012", "8",   "60 M",   "15.3%",  "ReLU, Dropout, GPU training"],
        ["VGG-16",     "2014", "16",  "138 M",  "7.3%",   "Deep stacks of 3×3 convolutions"],
        ["GoogLeNet",  "2014", "22",  "6.8 M",  "6.7%",   "Inception module + 1×1 bottleneck"],
        ["ResNet-50",  "2016", "50",  "25 M",   "3.57%",  "Residual (skip) connections"],
        ["MobileNetV2","2018", "53",  "3.4 M",  "~5%",    "Depthwise separable convolutions"],
        ["EfficientB0","2019", "—",   "5.3 M",  "2.9%",   "Compound depth/width/res scaling"],
    ]

    available = PAGE_W - 2 * MARGIN
    col_widths = [58, 26, 28, 34, 42, None]
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
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    return t


def build_story(styles):
    s = styles
    story = []

    # ---------- Header ----------
    story.append(Paragraph("Convolutional Neural Networks: A Survey", s["title"]))
    story.append(Paragraph(
        "Architecture, Code Examples, Classic Networks, and Recent Trends  ·  March 2026",
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
        "provides a minimal PyTorch implementation, compares landmark architectures, and "
        "discusses current research directions.",
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

    # ---------- 4. Classic Networks ----------
    story.append(Paragraph("4. Classic Network Comparison", s["h1"]))
    story.append(Paragraph(
        "Table 1 traces landmark CNN milestones from LeNet-5 to EfficientNet, showing the trend "
        "toward deeper networks, residual connections, and mobile-friendly architectures.",
        s["body"],
    ))
    story.append(classic_table())
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Table 1. Landmark CNN architectures. Top-5 error on ImageNet ILSVRC (where reported).",
        s["caption"],
    ))

    # ---------- 5. Recent Trends ----------
    story.append(Paragraph("5. Recent Trends", s["h1"]))
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

    # ---------- 6. Conclusion ----------
    story.append(Paragraph("6. Conclusion", s["h1"]))
    story.append(Paragraph(
        "CNNs have transformed computer vision, evolving from LeNet's digit recogniser to "
        "architectures achieving superhuman performance. Hybrid CNN–Transformer models and "
        "self-supervised pre-training continue to push accuracy–efficiency frontiers, ensuring "
        "CNNs remain central to AI progress.",
        s["body"],
    ))

    # ---------- References ----------
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.grey, spaceBefore=5, spaceAfter=3))
    story.append(Paragraph("References", s["h1"]))
    refs = [
        "[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with "
        "Deep Convolutional Neural Networks. <i>NeurIPS</i>, 25.",
        "[2] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning "
        "applied to document recognition. <i>Proc. IEEE</i>, 86(11), 2278–2324.",
        "[3] Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for "
        "Large-Scale Image Recognition. <i>arXiv:1409.1556</i>.",
        "[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image "
        "Recognition. <i>CVPR</i>, 770–778.",
        "[5] Szegedy, C., et al. (2015). Going Deeper with Convolutions. <i>CVPR</i>, 1–9.",
        "[6] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. "
        "<i>ICML</i>, 6105–6114.",
        "[7] Dosovitskiy, A., et al. (2020). An Image is Worth 16×16 Words: Transformers for "
        "Image Recognition at Scale. <i>arXiv:2010.11929</i>.",
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
