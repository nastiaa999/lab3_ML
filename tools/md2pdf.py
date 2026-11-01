"""Простая конвертация PLAN.md (заголовки, абзацы, списки, таблицы, **жирный**) в PDF."""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

ACCENT = colors.HexColor("#1f4e79")
base = ParagraphStyle("base", fontName="Arial", fontSize=10, leading=14, alignment=TA_LEFT)
cell = ParagraphStyle("cell", parent=base, fontSize=8.8, leading=11.5)
cell_head = ParagraphStyle("cellh", parent=cell, fontName="Arial-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=17, leading=22,
                    textColor=ACCENT, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=13, leading=17,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=5)


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def make_table(rows, width):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    data = [[Paragraph(inline(c), cell_head if i == 0 else cell) for c in r]
            for i, r in enumerate(cells)]
    ncol = len(cells[0])
    lens = [max(len(r[j]) for r in cells) for j in range(ncol)]
    longest_word = [max(len(w) for r in cells for w in r[j].split() or [""]) for j in range(ncol)]
    weights = [min(max(l, 2.0 * lw, 6), 60) for l, lw in zip(lens, longest_word)]
    col_w = [width * w / sum(weights) for w in weights]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bccb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build(src, dst):
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="План ЛР3")
    lines = open(src, encoding="utf-8").read().splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            story += [make_table(rows, doc.width), Spacer(1, 6)]
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(ListItem(Paragraph(inline(lines[i][2:]), base), leftIndent=12))
                i += 1
            story += [ListFlowable(items, bulletType="bullet", start="•", leftIndent=12),
                      Spacer(1, 4)]
            continue
        elif line.strip():
            story += [Paragraph(inline(line), base), Spacer(1, 4)]
        i += 1

    def footer(canvas, d):
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"стр. {d.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
