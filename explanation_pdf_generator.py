# explanation_pdf_generator.py
# DockExplain — generates a formatted PDF from the Gemma 4 explanation JSON.
#
# Run standalone:
#   python explanation_pdf_generator.py \
#       ../data/results/erlotinib/erlotinib_explanation.json
#
# Or imported automatically by run_dockexplain.py as the final step.

import json, os, re, sys
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable

# ── Colour palette ────────────────────────────────────────────────────────
C_NAVY      = colors.HexColor("#1A2E4A")
C_TEAL      = colors.HexColor("#1B7A6E")
C_LIGHT_BG  = colors.HexColor("#F4F7FA")
C_WARN_BG   = colors.HexColor("#FFF8E1")
C_WARN_LINE = colors.HexColor("#F9A825")
C_TEXT      = colors.HexColor("#1C1C1C")
C_SUBTEXT   = colors.HexColor("#4A4A4A")
C_WHITE     = colors.white
C_RULE      = colors.HexColor("#D0D8E4")
C_TABLE_ALT = colors.HexColor("#EDF2F7")


# ════════════════════════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════════════════════════

def _build_styles():
    return {
        "section_h1": ParagraphStyle("section_h1",
            fontName="Helvetica-Bold", fontSize=15, textColor=C_NAVY,
            leading=20, spaceBefore=18, spaceAfter=6),
        "section_h2": ParagraphStyle("section_h2",
            fontName="Helvetica-Bold", fontSize=12, textColor=C_TEAL,
            leading=16, spaceBefore=12, spaceAfter=4),
        "section_h3": ParagraphStyle("section_h3",
            fontName="Helvetica-BoldOblique", fontSize=11, textColor=C_NAVY,
            leading=14, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=10, textColor=C_TEXT,
            leading=15, alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet",
            fontName="Helvetica", fontSize=10, textColor=C_TEXT,
            leading=14, leftIndent=16, spaceAfter=3),
        "info_label": ParagraphStyle("info_label",
            fontName="Helvetica-Bold", fontSize=9, textColor=C_SUBTEXT,
            leading=12),
        "info_value": ParagraphStyle("info_value",
            fontName="Helvetica", fontSize=10, textColor=C_TEXT,
            leading=14, spaceAfter=2),
        "table_header": ParagraphStyle("table_header",
            fontName="Helvetica-Bold", fontSize=9, textColor=C_WHITE,
            leading=12, alignment=TA_CENTER),
        "table_cell": ParagraphStyle("table_cell",
            fontName="Helvetica", fontSize=9, textColor=C_TEXT, leading=12),
        "bib_entry": ParagraphStyle("bib_entry",
            fontName="Helvetica", fontSize=9, textColor=C_SUBTEXT,
            leading=13, leftIndent=20, firstLineIndent=-20, spaceAfter=6),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8, textColor=C_SUBTEXT,
            leading=11, alignment=TA_CENTER, spaceAfter=4),
        "warn_text": ParagraphStyle("warn_text",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#5D4037"), leading=13),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION BAR FLOWABLE
# ════════════════════════════════════════════════════════════════════════════

class SectionBar(Flowable):
    """Teal left bar + bold section title."""
    def __init__(self, text, width):
        super().__init__()
        self._text  = text
        self._width = width
        self.height = 22

    def draw(self):
        c = self.canv
        c.setFillColor(C_TEAL)
        c.rect(0, 2, 4, 16, fill=1, stroke=0)
        c.setFillColor(C_NAVY)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(10, 5, self._text)
        c.setStrokeColor(C_RULE)
        c.setLineWidth(0.5)
        c.line(0, 1, self._width, 1)


# ════════════════════════════════════════════════════════════════════════════
# COVER PAGE — drawn via canvas callback (NOT a Flowable)
# Using a Flowable for a full-page cover causes LayoutError because
# ReportLab tries to fit it inside the content frame.
# Drawing on the canvas directly bypasses the frame entirely.
# ════════════════════════════════════════════════════════════════════════════

class _CoverData:
    def __init__(self, drug, protein, score_str, mode,
                 n_interactions, n_papers, generated):
        self.drug           = drug
        self.protein        = protein
        self.score_str      = score_str
        self.mode           = mode
        self.n_interactions = n_interactions
        self.n_papers       = n_papers
        self.generated      = generated


def _draw_cover(canvas, cover: _CoverData):
    """Paint the cover directly onto the canvas at page 1."""
    W, H = A4

    # Navy background
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Teal accent bars
    canvas.setFillColor(C_TEAL)
    canvas.rect(0, H - 0.8*cm, W, 0.8*cm, fill=1, stroke=0)
    canvas.rect(0, 0, W, 0.6*cm, fill=1, stroke=0)

    # Brand
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawCentredString(W/2, H - 2.5*cm, "DockExplain")
    canvas.setFillColor(colors.HexColor("#B0C4D8"))
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(W/2, H - 3.3*cm,
        "AI-Powered Molecular Docking Explanation Report")

    # Rule
    canvas.setStrokeColor(C_TEAL)
    canvas.setLineWidth(1.5)
    canvas.line(2*cm, H - 3.9*cm, W - 2*cm, H - 3.9*cm)

    # Drug name
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 22)
    drug_d = cover.drug if len(cover.drug) <= 34 else cover.drug[:31] + "..."
    canvas.drawCentredString(W/2, H - 5.5*cm, drug_d)

    canvas.setFillColor(colors.HexColor("#B0C4D8"))
    canvas.setFont("Helvetica", 13)
    canvas.drawCentredString(W/2, H - 6.4*cm, "binding to")

    # Protein name (word-wrap up to 2 lines)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 16)
    prot = cover.protein[:70]
    words = prot.split()
    line1, line2 = [], []
    for w in words:
        if len(" ".join(line1 + [w])) <= 42:
            line1.append(w)
        else:
            line2.append(w)
    canvas.drawCentredString(W/2, H - 7.5*cm, " ".join(line1))
    if line2:
        canvas.drawCentredString(W/2, H - 8.4*cm, " ".join(line2))

    # Stat cards
    card_y, card_h, card_w = H - 12.0*cm, 2.5*cm, 4.2*cm
    xs    = [1.5*cm, 6.1*cm, 10.7*cm, 15.3*cm]
    cards = [
        ("Binding Score",  cover.score_str),
        ("Pipeline Mode",  f"Mode {cover.mode}"),
        ("Interactions",   str(cover.n_interactions)),
        ("Papers Cited",   str(cover.n_papers)),
    ]
    for (label, value), cx in zip(cards, xs):
        canvas.setFillColor(colors.HexColor("#243B55"))
        canvas.roundRect(cx, card_y, card_w, card_h, 4, fill=1, stroke=0)
        canvas.setFillColor(C_TEAL)
        canvas.roundRect(cx, card_y + card_h - 0.3*cm,
                         card_w, 0.3*cm, 2, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#8AADC4"))
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(cx + card_w/2,
                                 card_y + card_h - 0.7*cm, label)
        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        val = value if len(str(value)) < 18 else str(value)[:15] + "..."
        canvas.drawCentredString(cx + card_w/2, card_y + 0.7*cm, val)

    # Timestamp
    canvas.setFillColor(colors.HexColor("#7A9BBF"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(W/2, 1.5*cm,
        f"Generated by DockExplain  •  {cover.generated}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE HEADER / FOOTER
# ════════════════════════════════════════════════════════════════════════════

def _make_callbacks(drug: str, protein: str, cover: _CoverData):
    def on_first_page(canvas, doc):
        _draw_cover(canvas, cover)
        # No header/footer on cover — it's full bleed

    def on_later_pages(canvas, doc):
        W, H = A4
        # Header
        canvas.saveState()
        canvas.setStrokeColor(C_TEAL)
        canvas.setLineWidth(1)
        canvas.line(1.5*cm, H - 1.2*cm, W - 1.5*cm, H - 1.2*cm)
        canvas.setFillColor(C_NAVY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(1.5*cm, H - 1.0*cm, "DockExplain")
        canvas.setFillColor(C_SUBTEXT)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 1.5*cm, H - 1.0*cm,
            "Molecular Docking Explanation Report")
        # Footer
        canvas.setStrokeColor(C_RULE)
        canvas.setLineWidth(0.5)
        canvas.line(1.5*cm, 1.5*cm, W - 1.5*cm, 1.5*cm)
        canvas.setFillColor(C_SUBTEXT)
        canvas.setFont("Helvetica", 7.5)
        label = f"{drug}  ↔  {protein}"
        if len(label) > 70:
            label = label[:67] + "..."
        canvas.drawString(1.5*cm, 1.1*cm, label)
        canvas.drawRightString(W - 1.5*cm, 1.1*cm, f"Page {doc.page}")
        canvas.setFillColor(C_TEAL)
        canvas.drawCentredString(W/2, 1.1*cm, "DockExplain")
        canvas.restoreState()

    return on_first_page, on_later_pages


# ════════════════════════════════════════════════════════════════════════════
# HELPER BUILDERS
# ════════════════════════════════════════════════════════════════════════════

# EXPLANATION_PDF_GENERATOR FIX
# Replace the _safe() function (lines 252-267) with this version
# This removes LaTeX formatting that Gemma 4 sometimes outputs

def _safe(text: str) -> str:
    """
    Escape XML, convert Markdown + LaTeX to ReportLab tags.
    
    Handles:
      - LaTeX math mode: $\text{...}$, $\alpha$, $\beta$, etc.
      - XML escaping: &, <, >
      - Markdown: **bold**, *italic*, `code`
      - Unicode: subscripts and superscripts
      - Links: markdown and wiki-style
    """
    import re
    
    # ── 1. LATEX CLEANUP (do this first, before XML escaping)
    # Remove LaTeX \text{...} formatting: ($\text{C}$) → (C)
    text = re.sub(r'\(\$\\text\{([^}]+)\}\$\)', r'(\1)', text)
    text = re.sub(r'\$\\text\{([^}]+)\}\$', r'\1', text)
    
    # Convert Greek letters to Unicode
    greek_letters = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
        r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
        r'\nu': 'ν', r'\xi': 'ξ', r'\omicron': 'ο', r'\pi': 'π',
        r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ',
        r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    }
    
    for latex_char, unicode_char in greek_letters.items():
        # Handle both $\alpha$ and \alpha forms
        text = re.sub(r'\$' + re.escape(latex_char) + r'\$', unicode_char, text)
        text = re.sub(re.escape(latex_char), unicode_char, text)
    
    # Remove any remaining LaTeX math delimiters with content
    # This handles edge cases like \(...\) or $...$ that weren't matched above
    text = re.sub(r'\\\(([^)]*)\\\)', r'\1', text)  # \(...\) → content
    text = re.sub(r'\$([^$]*)\$', r'\1', text)       # $...$ → content
    
    # ── 2. XML ESCAPING (AFTER LaTeX cleanup to avoid escaping our Greek letters)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # ── 3. MARKDOWN CONVERSION
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`',
                  r'<font face="Courier" size="9">\1</font>', text)
    
    # ── 4. UNICODE SUB/SUPERSCRIPTS
    unicode_replacements = [
        ("₀","<sub>0</sub>"), ("₁","<sub>1</sub>"), ("₂","<sub>2</sub>"),
        ("₃","<sub>3</sub>"), ("₄","<sub>4</sub>"), ("₅","<sub>5</sub>"),
        ("₆","<sub>6</sub>"), ("₇","<sub>7</sub>"), ("₈","<sub>8</sub>"),
        ("₉","<sub>9</sub>"),
        ("⁰","<super>0</super>"), ("¹","<super>1</super>"), ("²","<super>2</super>"),
        ("³","<super>3</super>"), ("⁴","<super>4</super>"), ("⁵","<super>5</super>"),
        ("⁶","<super>6</super>"), ("⁷","<super>7</super>"), ("⁸","<super>8</super>"),
        ("⁹","<super>9</super>"),
        ("⁺","<super>+</super>"), ("⁻","<super>-</super>"),
    ]
    
    for char, tag in unicode_replacements:
        text = text.replace(char, tag)
    
    # ── 5. MARKDOWN LINKS (strip URLs, keep text)
    # Handle [[N](...)] style (numbered references)
    text = re.sub(r'\[\[(\d+)\]\([^)]+\)\]', r'[\1]', text)
    # Handle [text](url) style
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    return text


def _info_table(rows, styles):
    data = [[Paragraph(lbl, styles["info_label"]),
             Paragraph(_safe(str(val)), styles["info_value"])]
            for lbl, val in rows]
    t = Table(data, colWidths=[4*cm, 12*cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_LIGHT_BG, C_WHITE]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("BOX",           (0,0), (-1,-1), 0.5, C_RULE),
    ]))
    return t


def _interaction_table(interactions, styles):
    headers = ["Residue", "Interaction Type", "Description"]
    data = [[Paragraph(h, styles["table_header"]) for h in headers]]
    for ix in interactions:
        data.append([
            Paragraph(ix.get("residue",""),           styles["table_cell"]),
            Paragraph(ix.get("interaction_type",""),  styles["table_cell"]),
            Paragraph(ix.get("description",""),       styles["table_cell"]),
        ])
    t = Table(data, colWidths=[3.5*cm, 4.5*cm, 9*cm],
              hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_TEAL),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_TABLE_ALT]),
        ("GRID",          (0,0), (-1,-1), 0.4, C_RULE),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


def _warn_box(text, styles, page_w):
    t = Table([[Paragraph(_safe(text), styles["warn_text"])]],
              colWidths=[page_w], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_WARN_BG),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING",(0,0), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("BOX",         (0,0), (-1,-1), 0.5, C_WARN_LINE),
        ("LINEBEFORE",  (0,0), (0,-1),  4,   C_WARN_LINE),
    ]))
    return t


def _parse_explanation(text: str, styles: dict) -> list:
    """Convert Gemma Markdown to ReportLab flowables."""
    flowables = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flowables.append(Spacer(1, 4)); i += 1; continue

        if line.startswith("## "):
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(_safe(line[3:]), styles["section_h1"]))
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                         color=C_RULE, spaceAfter=4))
            i += 1; continue

        if line.startswith("### "):
            flowables.append(Paragraph(_safe(line[4:]), styles["section_h2"]))
            i += 1; continue

        if line.startswith("#### "):
            flowables.append(Paragraph(_safe(line[5:]), styles["section_h3"]))
            i += 1; continue

        if re.match(r'^[\*\-•]\s+', line):
            content = re.sub(r'^[\*\-•]\s+', '', line)
            flowables.append(Paragraph("• " + _safe(content), styles["bullet"]))
            i += 1; continue

        if re.match(r'^\d+\.\s+', line):
            num     = re.match(r'^(\d+)\.', line).group(1)
            content = re.sub(r'^\d+\.\s+', '', line)
            flowables.append(Paragraph(
                f"<b>{num}.</b> " + _safe(content), styles["body"]))
            i += 1; continue

        if re.match(r'^[-─═\*]{3,}$', line):
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                         color=C_RULE, spaceAfter=4))
            i += 1; continue

        # Accumulate paragraph lines
        para_lines = [line]; i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if (not nxt.strip()
                    or nxt.startswith("#")
                    or re.match(r'^[\*\-•]\s+', nxt)
                    or re.match(r'^\d+\.\s+', nxt)
                    or re.match(r'^[-─═\*]{3,}$', nxt)):
                break
            para_lines.append(nxt); i += 1

        para = " ".join(para_lines).strip()
        if para:
            flowables.append(Paragraph(_safe(para), styles["body"]))

    return flowables


# ════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_explanation_pdf(
    explanation_json_path: str,
    out_pdf_path: str | None = None,
) -> str:
    """
    Generate a formatted PDF report from a DockExplain explanation JSON.

    Args:
        explanation_json_path : path to *_explanation.json
        out_pdf_path          : output PDF path (default: same dir, .pdf)

    Returns:
        Path to the generated PDF.
    """
    with open(explanation_json_path) as f:
        data = json.load(f)

    if out_pdf_path is None:
        out_pdf_path = explanation_json_path.replace(".json", ".pdf")

    # Load companion results JSON for interaction table
    results_path = explanation_json_path.replace(
        "_explanation.json", "_results.json"
    )
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    # Extract fields
    drug           = data.get("drug", "Unknown Drug")
    protein        = data.get("protein", results.get("protein", "Unknown Protein"))
    explanation    = data.get("explanation", "")
    bibliography   = data.get("bibliography", "")
    n_interactions = data.get("n_interactions", 0)
    n_papers       = data.get("n_papers", 0)
    multi_role     = data.get("multi_role", [])
    sources        = data.get("sources", [])

    score          = results.get("docking_score")
    mode           = results.get("mode", "?")
    interactions   = results.get("interactions", [])
    drug_smiles    = results.get("drug_smiles", "")

    score_str  = (f"{score:.2f} kcal/mol" if score is not None
                  else "Crystal pose")
    generated  = datetime.now().strftime("%d %B %Y, %H:%M")

    # Document setup
    W, H     = A4
    margin   = 1.8*cm
    page_w   = W - 2 * margin

    doc = SimpleDocTemplate(
        out_pdf_path, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=2.0*cm, bottomMargin=2.2*cm,
        title=f"DockExplain — {drug} binding to {protein}",
        author="DockExplain AI",
        subject="Molecular Docking Explanation Report",
    )

    styles   = _build_styles()
    cover    = _CoverData(drug, protein, score_str, mode,
                          n_interactions, n_papers, generated)
    first_cb, later_cb = _make_callbacks(drug, protein, cover)

    story = []

    # ── Page 1: Cover (drawn by callback; PageBreak advances to p.2) ────
    story.append(PageBreak())

    # ── Page 2: Docking Summary ──────────────────────────────────────────
    story.append(SectionBar("1. Docking Summary", page_w))
    story.append(Spacer(1, 10))

    mode_labels = {
        "A": "Mode A — Crystal ligand (no docking)",
        "B": "Mode B — New drug docked into crystal site",
        "C": "Mode C — Blind docking (apo structure)",
    }
    info_rows = [
        ("Drug",             drug),
        ("Protein / Target", protein),
        ("Binding Score",    score_str),
        ("Pipeline Mode",    mode_labels.get(mode, mode)),
        ("Interactions",     f"{n_interactions} detected by ProLIF"),
    ]
    if drug_smiles:
        smiles_d = drug_smiles[:60] + ("..." if len(drug_smiles) > 60 else "")
        info_rows.append(("SMILES", smiles_d))
    if multi_role:
        info_rows.append(("Multi-role Residues", ", ".join(multi_role)))

    story.append(_info_table(info_rows, styles))
    story.append(Spacer(1, 14))

    if interactions:
        story.append(Paragraph("ProLIF Interaction Fingerprint",
                               styles["section_h2"]))
        story.append(Spacer(1, 4))
        story.append(_interaction_table(interactions, styles))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Table 1. All {n_interactions} interactions between "
            f"{drug} and {protein} detected by ProLIF fingerprinting.",
            styles["caption"]))

    story.append(PageBreak())

    # ── Page 3+: Explanation ─────────────────────────────────────────────
    story.append(SectionBar("2. AI-Generated Explanation (Gemma 4)",
                             page_w))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The explanation below was generated by Gemma 4 using the ProLIF "
        "interaction fingerprint, PyMOL visualization context, and literature "
        "evidence gathered from PubMed, PubChem, UniProt, and ChEMBL.",
        styles["body"]))
    story.append(Spacer(1, 8))
    story.extend(_parse_explanation(explanation, styles))

    # ── Bibliography ─────────────────────────────────────────────────────
    if bibliography or sources:
        story.append(PageBreak())
        story.append(SectionBar("3. Bibliography", page_w))
        story.append(Spacer(1, 8))

        if bibliography:
            for line in bibliography.split("\n"):
                line = line.strip()
                if not line or line.startswith("=") or line == "BIBLIOGRAPHY":
                    continue
                if line.startswith("Peer-reviewed") or line.startswith("Web sources"):
                    story.append(Paragraph(line, styles["section_h2"]))
                    continue
                m = re.match(r'^\[(\d+)\]\s+(.+)', line)
                if m:
                    story.append(Paragraph(
                        f"<b>[{m.group(1)}]</b> {_safe(m.group(2))}",
                        styles["bib_entry"]))
                elif line.startswith("PMID") or line.startswith("http"):
                    story.append(Paragraph(
                        f"    {_safe(line)}", styles["bib_entry"]))
                elif line:
                    story.append(Paragraph(_safe(line), styles["bib_entry"]))
        elif sources:
            story.append(Paragraph("Sources", styles["section_h2"]))
            for i, url in enumerate(sources[:20], 1):
                story.append(Paragraph(
                    f"[{i}] {_safe(url)}", styles["bib_entry"]))

    # ── Disclaimer ───────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(_warn_box(
        "<b>Disclaimer:</b> This report was generated automatically by "
        "DockExplain using Gemma 4 (local LLM). Results are based on "
        "AutoDock Vina docking and ProLIF interaction fingerprinting at "
        "pH 7.4 (physiological standard). Intended for research purposes "
        "only — not a substitute for experimental validation or expert "
        "biochemical interpretation.",
        styles, page_w))

    # ── Build ────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=first_cb, onLaterPages=later_cb)
    print(f"  ✓ PDF report saved: {out_pdf_path}")
    return out_pdf_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="DockExplain — generate PDF from explanation JSON")
    parser.add_argument("json", nargs="?",
        default="../data/results/erlotinib/erlotinib_explanation.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found")
        print("Run gemma_explainer.py first")
        sys.exit(1)

    generate_explanation_pdf(args.json, args.out)