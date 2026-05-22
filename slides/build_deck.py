"""
Build the "Phishing Website Detection" presentation (.pptx) for CSSE 415.

Design: dark background, minimal — no boxes/cards, single light text color,
short speaking-point bullets. Color lives only in the chart/QR images.

Real numbers come from notebooks/dataset1.csv + the team's progress reports.
QR code: set FIREBASE_URL below once the demo is hosted, then re-run.
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qrcode

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
FIREBASE_URL = None  # <-- paste the hosted demo URL here, then re-run to embed a real QR

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)
OUT_PPTX = os.environ.get("DECK_OUT", os.path.join(HERE, "Phishing_Website_Detection.pptx"))

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
BG    = "#0B1020"   # slide background
TEXT  = "#ECEFF4"   # the single basic text color
MUTED = "#9AA6BC"   # captions / secondary
ACC   = "#3B82F6"   # one thin underline accent only
# chart-only colors
GREEN = "#2EE6A0"; BLUE = "#3B82F6"; CYAN = "#22D3EE"; GRAY = "#566077"
RED   = "#F87171"; AMBER = "#FBBF24"

def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.lstrip("#"))

# ============================================================================
# 1. CHART ASSETS (matplotlib, dark theme) — color is allowed here
# ============================================================================
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": "#2A3550",
    "xtick.color": TEXT, "ytick.color": TEXT, "font.family": "DejaVu Sans",
    "font.size": 12,
})

def chart_model_accuracy():
    labels = ["Baseline", "kNN", "Ridge", "Decision\nTree", "Logistic\nReg.",
              "SVM", "XGBoost", "Random\nForest", "Gradient\nBoost"]
    vals   = [0.557, 0.949, 0.951, 0.958, 0.962, 0.974, 0.974, 0.975, 0.977]
    colors = [GRAY, BLUE, BLUE, BLUE, BLUE, GREEN, GREEN, GREEN, GREEN]
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.72, zorder=3)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Test accuracy")
    ax.yaxis.grid(True, color="#222D45", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10.5)
    ax.tick_params(axis="y", length=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)
    ax.axhline(0.557, color=GRAY, ls="--", lw=1, zorder=2)
    ax.text(8.4, 0.566, "majority-class baseline 55.7%", ha="right", va="bottom",
            fontsize=9, color=MUTED, style="italic")
    fig.tight_layout()
    p = os.path.join(ASSETS, "model_accuracy.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_top4_acc_recall():
    models = ["Gradient\nBoost", "Random\nForest", "SVM", "XGBoost"]
    acc    = [0.977, 0.975, 0.974, 0.974]
    recall = [0.988, 0.980, 0.990, 0.985]  # recall on the PHISHING class
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    b1 = ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=BLUE, zorder=3)
    b2 = ax.bar([i + w/2 for i in x], recall, w, label="Phishing recall", color=GREEN, zorder=3)
    ax.set_ylim(0.9, 1.005)
    ax.set_xticks(list(x)); ax.set_xticklabels(models, fontsize=11)
    ax.yaxis.grid(True, color="#222D45", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                    f"{b.get_height()*100:.1f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT)
    leg = ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=11,
                    bbox_to_anchor=(0.5, -0.22))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    p = os.path.join(ASSETS, "top4_acc_recall.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_confusion():
    cm = [[945, 35], [15, 1216]]  # Gradient Boosting — best model
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    cell_colors = [[GREEN, RED], [RED, GREEN]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         color=cell_colors[i][j], alpha=0.85, zorder=1))
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center",
                    fontsize=24, fontweight="bold", color="#0B1020", zorder=2)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Phishing"], fontsize=12)
    ax.set_yticklabels(["Legit", "Phishing"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    p = os.path.join(ASSETS, "confusion_matrix.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_feature_importance():
    feats = ["SSL final state", "URL of anchor (<a>)", "Web traffic rank",
             "Sub-domain depth", "Links in tags", "Prefix/suffix '-'",
             "Server form handler", "Links pointing to page"]
    imp   = [0.334, 0.236, 0.074, 0.067, 0.043, 0.040, 0.020, 0.019]
    feats = feats[::-1]; imp = imp[::-1]
    colors = [GREEN if v >= 0.2 else BLUE for v in imp]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.barh(feats, imp, color=colors, zorder=3)
    ax.set_xlabel("Random Forest importance")
    ax.xaxis.grid(True, color="#222D45", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=11)
    for b, v in zip(bars, imp):
        ax.text(v + 0.005, b.get_y()+b.get_height()/2, f"{v:.2f}",
                va="center", fontsize=10, color=TEXT)
    ax.set_xlim(0, 0.39)
    fig.tight_layout()
    p = os.path.join(ASSETS, "feature_importance.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_class_balance():
    fig, ax = plt.subplots(figsize=(5.0, 4.7))
    sizes = [6157, 4898]; cols = [RED, GREEN]
    wedges, _ = ax.pie(sizes, colors=cols, startangle=90, radius=1.0,
                       wedgeprops=dict(width=0.42, edgecolor=BG, linewidth=3))
    ax.text(0, 0, "11,055\nsites", ha="center", va="center",
            fontsize=20, fontweight="bold", color=TEXT)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    leg = ax.legend(wedges, ["Phishing — 6,157  (55.7%)", "Legitimate — 4,898  (44.3%)"],
                    loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False,
                    fontsize=12.5, handlelength=1.1, handleheight=1.1, labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)
    p = os.path.join(ASSETS, "class_balance.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_pipeline():
    from matplotlib.patches import FancyBboxPatch
    steps = [
        ("1", "Clean &\nverify"),
        ("2", "Encode\nfeatures"),
        ("3", "80/20\nsplit"),
        ("4", "Scale\nwhere needed"),
        ("5", "Feature\nengineering"),
        ("6", "Train &\ncompare"),
    ]
    accents = [GREEN, BLUE, CYAN, GREEN, BLUE, GREEN]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(12.8, 2.9))
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.axis("off")
    bw, bh, y = 0.80, 0.74, 0.13
    for i, (num, label) in enumerate(steps):
        cx = i + 0.5; x = cx - bw/2
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=2.2, edgecolor=accents[i], facecolor="#141C30"))
        ax.text(cx, y + bh - 0.16, num, ha="center", va="center",
                fontsize=16, fontweight="bold", color=accents[i])
        ax.text(cx, y + bh*0.36, label, ha="center", va="center",
                fontsize=13, color=TEXT, linespacing=1.15)
        if i < n - 1:
            ax.annotate("", xy=(i + 1 + 0.5 - bw/2 - 0.015, y + bh/2),
                        xytext=(x + bw + 0.015, y + bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    p = os.path.join(ASSETS, "pipeline.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_svm_confusion():
    cm = [[938, 42], [16, 1215]]  # SVM {C:10, gamma:0.1}
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    cell_colors = [[GREEN, RED], [RED, GREEN]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         color=cell_colors[i][j], alpha=0.85, zorder=1))
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center",
                    fontsize=24, fontweight="bold", color="#0B1020", zorder=2)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Phishing"], fontsize=12)
    ax.set_yticklabels(["Legit", "Phishing"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    p = os.path.join(ASSETS, "svm_confusion.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def chart_svm_pr():
    classes = ["Legitimate", "Phishing"]
    precision = [0.98, 0.97]
    recall    = [0.96, 0.99]
    f1        = [0.97, 0.98]
    x = range(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    b1 = ax.bar([i - w for i in x], precision, w, label="Precision", color=BLUE, zorder=3)
    b2 = ax.bar([i      for i in x], recall,    w, label="Recall",    color=GREEN, zorder=3)
    b3 = ax.bar([i + w for i in x], f1,         w, label="F1",        color=CYAN, zorder=3)
    ax.set_ylim(0.9, 1.005)
    ax.set_xticks(list(x)); ax.set_xticklabels(classes, fontsize=12)
    ax.yaxis.grid(True, color="#222D45", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT)
    leg = ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=11,
                    bbox_to_anchor=(0.5, -0.22))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    p = os.path.join(ASSETS, "svm_pr.png")
    fig.savefig(p, dpi=200); plt.close(fig); return p

def make_qr():
    p = os.path.join(ASSETS, "qr.png")
    if FIREBASE_URL:
        qr = qrcode.QRCode(box_size=12, border=2,
                           error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(FIREBASE_URL); qr.make(fit=True)
        img = qr.make_image(fill_color="#0B1020", back_color="white")
        img.save(p); return p, True
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#0B1020", lw=3, ls="--"))
    ax.text(0.5, 0.58, "QR", ha="center", va="center", fontsize=46, fontweight="bold", color="#0B1020")
    ax.text(0.5, 0.33, "Firebase link\npending", ha="center", va="center", fontsize=13, color="#566077")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(p, dpi=200); plt.close(fig); return p, False

print("Rendering charts...")
IMG_ACC   = chart_model_accuracy()
IMG_TOP4  = chart_top4_acc_recall()
IMG_CM    = chart_confusion()
IMG_FEAT  = chart_feature_importance()
IMG_BAL   = chart_class_balance()
IMG_PIPE  = chart_pipeline()
IMG_SVMCM = chart_svm_confusion()
IMG_SVMPR = chart_svm_pr()
IMG_QR, QR_REAL = make_qr()

# ============================================================================
# 2. PPTX — minimal, no boxes, single text color
# ============================================================================
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height

def add_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    return s

def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

def textbox(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(2)
    return tb, tf

def para(tf, text, size, color=TEXT, bold=False, italic=False, first=False,
         align=PP_ALIGN.LEFT, space_after=10, space_before=0, name="Segoe UI"):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after); p.space_before = Pt(space_before)
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = rgb(color); r.font.name = name
    return p

def title_block(slide, title):
    _, tf = textbox(slide, 0.7, 0.5, 12.0, 1.0)
    para(tf, title, 32, TEXT, bold=True, first=True, space_after=0)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.76),
                                 Inches(1.32), Inches(1.7), Inches(0.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ACC); bar.line.fill.background()
    bar.shadow.inherit = False

def bullets(slide, l, t, w, h, items, size=20, gap=14):
    _, tf = textbox(slide, l, t, w, h)
    for i, it in enumerate(items):
        para(tf, "•   " + it, size, TEXT, first=(i == 0), space_after=gap)
    return tf

def pic(slide, path, l, t, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(l), Inches(t), **kw)

# ---- Slide 1: Title -------------------------------------------------------
s = add_slide()
_, tf = textbox(s, 0.9, 2.55, 11.5, 2.2)
para(tf, "Phishing Website Detection", 46, TEXT, bold=True, first=True, space_after=10)
para(tf, "Telling phishing sites from legitimate ones with machine learning", 18, MUTED, space_after=0)
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.96), Inches(4.5), Inches(2.6), Inches(0.05))
bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ACC); bar.line.fill.background(); bar.shadow.inherit = False
_, nf = textbox(s, 0.9, 4.75, 11.5, 1.2)
para(nf, "Erwin Perkowski   ·   Mark Joseph Kenny   ·   Sanil Maheshwari", 18, TEXT, bold=True, first=True, space_after=4)
para(nf, "CSSE 415 — Machine Learning · Final Project", 14, MUTED)
notes(s, "Title slide. Welcome the class with energy. The hook is on the next slide. "
         "Make sure all three of you are ready to own your sections.")

# ---- Slide 2: Hook --------------------------------------------------------
s = add_slide()
_, tf = textbox(s, 1.0, 2.3, 11.3, 3.0, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Ever been scared to click a link?", 40, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=22)
para(tf, "Raise your hand if you've ever hesitated.", 22, MUTED, align=PP_ALIGN.CENTER, space_after=0)
notes(s, "HOOK — first 30 seconds, engage the room. Ask the question, wait for hands. "
         "Say: that hesitation is the instinct we tried to teach a machine. Eye contact, speak up.")

# ---- Slide 3: Why phishing matters ---------------------------------------
s = add_slide()
title_block(s, "Why Phishing?")
bullets(s, 0.9, 2.0, 11.5, 4.8, [
    "Fake sites built to steal logins, money, and data",
    "They appear and disappear within hours",
    "Many look identical to the real thing",
    "Block-lists and fixed rules can't keep up",
    "The #1 entry point for breaches — billions lost each year",
], size=22, gap=18)
notes(s, "Cover the rubric out loud: WHAT (fake sites stealing credentials), WHY we care "
         "(huge losses — mention the FBI IC3 / BEC ~$2.9B figure and the ~$100M Google & Facebook invoice scam), "
         "WHY it's hard (fast-changing, look real, rules don't scale), and our angle: let ML learn the subtle "
         "signals instead of hard-coded rules. Erwin: tell your personal phishing-scandal story here.")

# ---- Slide 4: Dataset -----------------------------------------------------
s = add_slide()
title_block(s, "The Dataset")
bullets(s, 0.9, 2.0, 6.4, 4.6, [
    "UCI / UIUC phishing dataset",
    "11,055 sites · 30 features · binary label",
    "Values encoded as −1 / 0 / +1",
    "Clean, research-grade — no missing values",
    "Also re-checked on 2 more datasets",
], size=20, gap=16)
pic(s, IMG_BAL, 7.6, 1.95, w=5.0)
notes(s, "Describe the data: UCI/UIUC phishing set, 11,055 sites, 30 features, binary label, already clean. "
         "Classes reasonably balanced (donut) so accuracy is a fair starting metric. We ran the same pipeline on "
         "2 more datasets to confirm it generalizes.")

# ---- Slide 5: Intuitive features -----------------------------------------
s = add_slide()
title_block(s, "What Makes a URL Look Phishy?")
left = ["🔗  URL length", "🔒  HTTPS / SSL state", "⚠️  Suspicious symbols ( @  //  -  IP )"]
right = ["⏳  Domain age", "↪  Redirects", "🪟  iframes & pop-ups"]
_, tf = textbox(s, 0.95, 2.2, 6.0, 4.4)
for i, t in enumerate(left):
    para(tf, t, 24, TEXT, first=(i == 0), space_after=26)
_, tf = textbox(s, 7.0, 2.2, 5.6, 4.4)
for i, t in enumerate(right):
    para(tf, t, 24, TEXT, first=(i == 0), space_after=26)
notes(s, "Walk through the intuitive features so the audience builds a mental model: URL length, HTTPS/SSL, "
         "suspicious symbols (@, //, -, raw IP), domain age, redirects, iframes/pop-ups. Tie back to the hook — "
         "these are the cues a careful human looks for; we measure all 30 automatically.")

# ---- Slide 6: Approach / pipeline ----------------------------------------
s = add_slide()
title_block(s, "Our Approach")
pic(s, IMG_PIPE, 0.5, 2.75, w=12.33)
_, cap = textbox(s, 0.9, 5.85, 11.5, 0.6)
para(cap, "Same split and protocol for every model", 16, MUTED, first=True, align=PP_ALIGN.CENTER)
notes(s, "Describe the pipeline so it's clear exactly what we did: clean/verify, encode, 80/20 stratified split "
         "(random_state 42), scale only for distance/margin models, feature engineering with pairwise interactions "
         "(up to 495 features, Lasso pruned to 320), consistent protocol across all 8 models and datasets.")

# ---- Slide 7: Models tested ----------------------------------------------
s = add_slide()
title_block(s, "Models We Tested")
_, cap = textbox(s, 0.9, 1.55, 11.5, 0.5)
para(cap, "8 classifiers — every one beats the 55.7% baseline", 16, MUTED, first=True)
pic(s, IMG_ACC, 1.6, 2.15, w=10.1)
notes(s, "We tested 8 classifiers from class: Logistic Regression, Decision Tree, kNN, Ridge, Random Forest, "
         "Gradient Boosting, SVM, XGBoost. The chart shows best accuracy per family vs the 55.7% majority baseline. "
         "Everything clears it easily; the green bars (SVM, XGBoost, RF, Gradient Boosting) are our strongest.")

# ---- Slide 8: Strongest models 1/2 ---------------------------------------
def two_models(slide, a, b):
    _, tf = textbox(slide, 0.95, 2.2, 5.6, 4.3)
    para(tf, a[0], 26, TEXT, bold=True, first=True, space_after=14)
    for i, t in enumerate(a[1]):
        para(tf, "•   " + t, 20, TEXT, space_after=12)
    _, tf = textbox(slide, 7.0, 2.2, 5.6, 4.3)
    para(tf, b[0], 26, TEXT, bold=True, first=True, space_after=14)
    for i, t in enumerate(b[1]):
        para(tf, "•   " + t, 20, TEXT, space_after=12)

s = add_slide()
title_block(s, "Strongest Models  (1 / 2)")
two_models(s,
    ("Gradient Boosting", ["97.7% accuracy — best overall", "98.8% phishing recall", "Few phishing sites missed"]),
    ("Random Forest", ["97.5% accuracy", "Captures interactions natively", "Top tells: SSL state & anchor links"]))
notes(s, "Emphasize WHY these win. Gradient Boosting (97.7%) is best overall with ~98.8% phishing recall. "
         "Random Forest (97.5%) is essentially tied and very robust — it captures interactions natively, "
         "which is why hand-made interaction features barely helped it.")

# ---- Slide 9: Strongest models 2/2 ---------------------------------------
s = add_slide()
title_block(s, "Strongest Models  (2 / 2)")
two_models(s,
    ("SVM (RBF)", ["97.4% accuracy", "Highest phishing recall — 99%", "Recall matters most to us"]),
    ("XGBoost", ["97.4% accuracy", "Regularized gradient boosting", "On par with RF and SVM"]))
notes(s, "SVM (97.4%) had the HIGHEST phishing recall (~99%) — call this out, because missing a phishing site is "
         "the costly error. XGBoost (97.4%) matches RF/SVM. The top four are within ~0.3% of each other; pick on "
         "recall + interpretability.")

# ---- Slide 10: Results ----------------------------------------------------
s = add_slide()
title_block(s, "Results — Best Model")
pic(s, IMG_CM, 0.7, 1.95, w=4.6)
pic(s, IMG_TOP4, 5.5, 1.95, w=4.2)
bullets(s, 9.9, 2.5, 3.0, 3.6, [
    "97.7% accuracy",
    "98.8% phishing recall",
    "15 of 1,231 phishing missed",
    "~40 pts above baseline",
], size=16, gap=14)
notes(s, "Results in detail (rubric wants tables/graphs). Left: confusion matrix of the best model (Gradient Boosting) "
         "— only 15 phishing sites missed out of 1,231. Middle: accuracy vs phishing recall for the top four. "
         "Right: the headline numbers. Stress recall again.")

# ---- Slide: SVM detail ----------------------------------------------------
s = add_slide()
title_block(s, "SVM — Highest Phishing Recall")
pic(s, IMG_SVMCM, 0.6, 1.95, w=4.4)
_, cap = textbox(s, 0.6, 6.35, 4.4, 0.5)
para(cap, "Confusion matrix — SVM (test set)", 12.5, MUTED, first=True, align=PP_ALIGN.CENTER)
pic(s, IMG_SVMPR, 5.25, 1.95, w=4.5)
bullets(s, 10.0, 2.6, 3.0, 3.6, [
    "97.4% accuracy",
    "99% phishing recall",
    "Only 16 phishing missed",
    "{ C = 10, gamma = 0.1 }",
], size=16, gap=14)
notes(s, "SVM detail. Confusion matrix [[938, 42], [16, 1215]] — only 16 phishing sites missed. "
         "Per-class precision/recall/F1 on the right: phishing recall is 0.99, the best of any model, which is the "
         "metric we care most about (a missed phishing site is the costly mistake). Tuned with C=10, gamma=0.1 via "
         "cross-validation. Slightly lower accuracy than Gradient Boosting but the strongest recall.")

# ---- Slide 11: Feature importance ----------------------------------------
s = add_slide()
title_block(s, "What Drives the Prediction?")
pic(s, IMG_FEAT, 0.7, 2.0, w=7.7)
bullets(s, 8.8, 2.5, 4.0, 3.8, [
    "SSL state + anchor links ≈ 57% of importance",
    "Matches intuition: fake certs, deceptive links",
    "Interactions add extra signal",
], size=18, gap=18)
notes(s, "Real Random Forest importances from our data. SSL final state (0.33) and URL-of-anchor (0.24) together are "
         "~57% of importance — is the certificate legit, and where do the page's links actually point. Matches "
         "security intuition. The report also flagged useful interactions (SSL×RightClick, anchor×Iframe).")

# ---- Slide 12: Demo -------------------------------------------------------
s = add_slide()
title_block(s, "Live Demo")
pic(s, IMG_QR, 1.3, 2.2, w=3.4)
_, qf = textbox(s, 1.0, 5.8, 4.0, 0.6)
para(qf, ("Scan to try it" if QR_REAL else "QR — Firebase link pending"),
     14, MUTED, first=True, align=PP_ALIGN.CENTER)
_, tf = textbox(s, 5.6, 2.5, 7.0, 3.6)
steps = ["You pick a URL", "We extract 30 features live (+ Safe Browsing)",
         "The models vote: phishing or legitimate", "We show the verdict and the signals behind it"]
for i, t in enumerate(steps):
    para(tf, f"{i+1}.   {t}", 22, TEXT, first=(i == 0), space_after=18)
notes(s, "DEMO — make it interactive. Put the printed QR on the board, give the class 2-3 links and let THEM guess "
         "before we run it. Suggested order (from DEMO.txt): wikipedia.org (legit), Google's Safe-Browsing test page "
         "(flagged), then http://192.168.1.1@bit.ly/freegift?login=admin (many URL red flags at once). "
         ">>> Send the Firebase URL to embed the real QR. <<<")

# ---- Slide 13: What worked / what didn't ----------------------------------
s = add_slide()
title_block(s, "What Worked & What Didn't")
_, tf = textbox(s, 0.95, 2.05, 5.6, 4.5)
para(tf, "Worked", 24, TEXT, bold=True, first=True, space_after=14)
for t in ["Ensembles + SVM all hit ~97–98%", "Feature engineering lifted linear models",
          "High phishing recall", "Held up across datasets"]:
    para(tf, "•   " + t, 20, TEXT, space_after=12)
_, tf = textbox(s, 7.0, 2.05, 5.6, 4.5)
para(tf, "Didn't help", 24, TEXT, bold=True, first=True, space_after=14)
for t in ["PCA — models already fast", "Lasso pruning — no accuracy gain",
          "Interactions barely moved trees", "Simple rules alone"]:
    para(tf, "•   " + t, 20, TEXT, space_after=12)
notes(s, "Honest evaluation. Worked: ensembles + SVM, feature engineering for linear models, high recall, "
         "reproducibility. Didn't: PCA wasn't needed (models already fast; it hurt interpretability), Lasso pruned "
         "features without raising accuracy, interaction features barely helped trees. PCA was an experiment, not a final model.")

# ---- Slide 14: Obstacles / lessons ---------------------------------------
s = add_slide()
title_block(s, "Obstacles & Lessons")
bullets(s, 0.95, 2.1, 11.5, 4.6, [
    "Feature explosion (495) → used Lasso to prune & stay interpretable",
    "Big datasets slow to tune → limited dimensions, budgeted compute",
    "Accuracy isn't enough → focused on recall + confusion matrices",
], size=22, gap=24)
notes(s, "Three concrete obstacles + what we learned. 1) Interaction features blew up to 495 → Lasso/regularization "
         "to prune and keep it interpretable. 2) Dataset 2 too large for SVM/boosting to fully tune → dimensionality "
         "limits / PCA, deliberate compute budget. 3) Realizing accuracy isn't the target → we now evaluate with "
         "recall + confusion matrices. Each of us can take one.")

# ---- Slide 15: Future work (more time with the data/models) ---------------
s = add_slide()
title_block(s, "Future Work")
bullets(s, 0.95, 2.0, 11.6, 4.8, [
    "Deeper hyperparameter search we had to skip on the big datasets",
    "More thorough feature-interaction engineering and selection",
    "Proper cross-dataset training and testing for generalization",
    "Stack / ensemble our strongest models together",
    "Add explainability (e.g. SHAP) to see why a site is flagged",
    "Error analysis on misclassified sites to find weak spots",
], size=20, gap=15)
notes(s, "Future work, focused on doing MORE with the data and models given more time: fuller hyperparameter "
         "searches we skipped on the large datasets, deeper feature-interaction engineering/selection, proper "
         "cross-dataset train/test, stacking our best models, adding SHAP explainability, and error analysis on the "
         "misclassified sites to understand where the models break.")

# ---- Slide 16: Conclusion -------------------------------------------------
s = add_slide()
title_block(s, "Conclusion")
bullets(s, 0.95, 2.0, 11.6, 4.2, [
    "Classic ML detects phishing well — 97.7% accuracy, 98.8% recall",
    "Ensembles (GB / RF / XGB) and SVM win",
    "SSL state and link behavior are the biggest tells",
    "PCA unnecessary; recall matters most",
    "Beats the majority baseline by ~40 points",
], size=21, gap=15)
_, tf = textbox(s, 0.95, 6.4, 11.6, 0.8)
para(tf, "Thank you — questions?", 22, MUTED, bold=True, first=True)
notes(s, "Wrap up: classic ML solves this well (~97.7% accuracy, ~98.8% phishing recall). Winners: ensembles + SVM. "
         "SSL state & link behavior dominate. PCA wasn't needed; recall is the key metric. Thank the class and invite "
         "questions — handle them calmly and positively.")

prs.save(OUT_PPTX)
print(f"Saved: {OUT_PPTX}")
print(f"Slides: {len(prs.slides._sldIdLst)}  | QR real: {QR_REAL}")
