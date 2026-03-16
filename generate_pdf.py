from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=1*inch,
        rightMargin=1*inch,
        topMargin=1*inch,
        bottomMargin=1*inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Normal'],
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=4,
        fontName='Helvetica-Bold',
    )
    author_style = ParagraphStyle(
        'Author',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        fontName='Helvetica-Bold',
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )

    story = []

    story.append(Paragraph(
        "Convolutional Neural Networks: A Brief Survey",
        title_style
    ))
    story.append(Paragraph(
        "March 2026",
        author_style
    ))

    story.append(Paragraph("Abstract", section_style))
    story.append(Paragraph(
        "Convolutional Neural Networks (CNNs) have become the cornerstone of modern computer vision, "
        "achieving state-of-the-art results across a wide range of tasks including image classification, "
        "object detection, and semantic segmentation. This brief survey outlines the key architectural "
        "innovations, landmark models, and current trends in CNN research.",
        body_style
    ))

    story.append(Paragraph("1. Introduction", section_style))
    story.append(Paragraph(
        "A CNN is a deep learning model that exploits the spatial structure of images through "
        "local receptive fields and parameter sharing. The convolutional layer applies a set of "
        "learnable filters to produce feature maps, capturing local patterns such as edges and textures. "
        "Pooling layers then subsample these maps, providing translation invariance and reducing "
        "computational cost. The combination of convolution, activation functions (e.g., ReLU), "
        "and pooling enables CNNs to learn hierarchical representations from raw pixels.",
        body_style
    ))

    story.append(Paragraph("2. Landmark Architectures", section_style))
    story.append(Paragraph(
        "<b>LeNet-5</b> (LeCun et al., 1998) pioneered CNNs for handwritten digit recognition. "
        "<b>AlexNet</b> (Krizhevsky et al., 2012) demonstrated that deep CNNs trained on GPUs "
        "dramatically outperform hand-crafted features on ImageNet. <b>VGGNet</b> (Simonyan & Zisserman, 2014) "
        "showed that network depth, using small 3×3 filters, is a critical component. "
        "<b>GoogLeNet/Inception</b> (Szegedy et al., 2015) introduced the Inception module for "
        "multi-scale feature extraction with controlled parameter count. "
        "<b>ResNet</b> (He et al., 2016) introduced residual connections, enabling training of "
        "networks with hundreds of layers by alleviating the vanishing-gradient problem. "
        "<b>DenseNet</b> (Huang et al., 2017) extended this idea by connecting each layer to "
        "every subsequent layer, promoting feature reuse.",
        body_style
    ))

    story.append(Paragraph("3. Efficiency and Lightweight Models", section_style))
    story.append(Paragraph(
        "Deploying CNNs on resource-constrained devices motivated compact architectures. "
        "<b>MobileNet</b> (Howard et al., 2017) replaced standard convolutions with depthwise "
        "separable convolutions, drastically reducing parameters. <b>EfficientNet</b> (Tan & Le, 2019) "
        "proposed compound scaling of depth, width, and resolution, achieving superior accuracy–efficiency "
        "trade-offs. Techniques such as pruning, quantization, and knowledge distillation further "
        "compress CNNs for edge inference.",
        body_style
    ))

    story.append(Paragraph("4. Beyond Classification", section_style))
    story.append(Paragraph(
        "CNNs power a diverse set of vision tasks. <b>Object detection</b> frameworks such as "
        "Faster R-CNN and YOLO combine CNN backbones with region proposal or anchor-based heads. "
        "<b>Semantic segmentation</b> models like FCN and DeepLab use dilated convolutions and "
        "encoder–decoder structures to produce dense pixel-wise predictions. "
        "<b>Generative models</b>, including DCGANs, employ CNN-based generators and discriminators "
        "for image synthesis. CNNs have also been adapted for video understanding, medical imaging, "
        "and natural language processing tasks.",
        body_style
    ))

    story.append(Paragraph("5. Recent Trends", section_style))
    story.append(Paragraph(
        "Vision Transformers (ViT) have challenged the dominance of CNNs by modeling long-range "
        "dependencies via self-attention. Hybrid models that integrate convolutional inductive biases "
        "with attention mechanisms (e.g., ConvNeXt, CvT) have demonstrated competitive performance. "
        "Self-supervised and contrastive pre-training (e.g., SimCLR, MAE) reduce dependence on "
        "labeled data. Neural Architecture Search (NAS) automates the design of CNN topologies "
        "tailored to specific hardware budgets.",
        body_style
    ))

    story.append(Paragraph("6. Conclusion", section_style))
    story.append(Paragraph(
        "CNNs have fundamentally transformed computer vision over the past decade, evolving from "
        "simple handcrafted filter banks to sophisticated architectures capable of superhuman "
        "performance. Ongoing research continues to push the boundaries of efficiency, scalability, "
        "and generalization, ensuring that CNNs and their descendants remain central to the "
        "advancement of artificial intelligence.",
        body_style
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("References", section_style))
    refs = [
        "[1] LeCun et al. (1998). Gradient-based learning applied to document recognition. <i>Proc. IEEE</i>.",
        "[2] Krizhevsky et al. (2012). ImageNet classification with deep CNNs. <i>NeurIPS</i>.",
        "[3] He et al. (2016). Deep residual learning for image recognition. <i>CVPR</i>.",
        "[4] Howard et al. (2017). MobileNets: Efficient CNNs for mobile vision applications. <i>arXiv</i>.",
        "[5] Tan & Le (2019). EfficientNet: Rethinking model scaling for CNNs. <i>ICML</i>.",
    ]
    ref_style = ParagraphStyle(
        'Ref', parent=body_style, fontSize=9, leading=12, spaceAfter=2
    )
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story)
    print(f"PDF written to {filename}")

if __name__ == '__main__':
    build_pdf('cnn_survey.pdf')
