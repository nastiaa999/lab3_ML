"""PDF для подготовки к защите из markdown-файлов.

Поддерживает: # / ## / ### заголовки, абзацы, списки «- », таблицы «|», блоки ``` кода, врезки «> »,
**жирный**, `код в тексте` и вставку кода проекта с номерами строк: строка «@@code файл 10-25».
Запуск: python tools/prep_pdf.py выход.pdf часть1.md часть2.md ...
"""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
pdfmetrics.registerFont(TTFont("Courier", "C:/Windows/Fonts/cour.ttf"))

ACCENT = colors.HexColor("#1f4e79")
base = ParagraphStyle("base", fontName="Arial", fontSize=10, leading=14)
cell = ParagraphStyle("cell", parent=base, fontSize=8.6, leading=11)
cell_head = ParagraphStyle("cellh", parent=cell, fontName="Arial-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=17, leading=22, textColor=ACCENT,
                    spaceAfter=8)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=13, leading=17, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=5)
h3 = ParagraphStyle("h3", parent=base, fontName="Arial-Bold", fontSize=11, leading=15, spaceBefore=6,
                    spaceAfter=3)
code_st = ParagraphStyle("code", parent=base, fontName="Courier", fontSize=7.6, leading=9.4)
note_st = ParagraphStyle("note", parent=base, fontSize=9.6, leading=13.5)


def inline(text):
    """`код` обрабатываем отдельно, чтобы звёздочки внутри него не стали жирным."""
    out = []
    for k, part in enumerate(text.split("`")):
        part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if k % 2:
            out.append(f'<font name="Courier" color="#7a1f1f">{part}</font>')
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part))
    return "".join(out)


def code_block(lines, numbers=None):
    out = []
    for k, ln in enumerate(lines):
        esc = ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        esc = re.sub(r"^( +)", lambda m: "&nbsp;" * len(m.group(1)), esc)
        pad = f"{numbers[k]:>3}".replace(" ", "&nbsp;") if numbers else ""
        num = f'<font color="#8a8a8a">{pad}</font>&nbsp;&nbsp;' if numbers else ""
        out.append(num + (esc or "&nbsp;"))
    t = Table([[Paragraph("<br/>".join(out), code_st)]], colWidths=[None])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                           ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def make_table(rows, width):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    data = [[Paragraph(inline(c), cell_head if i == 0 else cell) for c in r] for i, r in enumerate(cells)]
    ncol = len(cells[0])
    lens = [max(len(r[j]) if j < len(r) else 0 for r in cells) for j in range(ncol)]
    longest = [max((len(w) for r in cells if j < len(r) for w in r[j].split()), default=1) for j in range(ncol)]
    weights = [min(max(l, 2.0 * lw, 6), 60) for l, lw in zip(lens, longest)]
    t = Table(data, colWidths=[width * w / sum(weights) for w in weights], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bccb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def note(lines, width):
    t = Table([[Paragraph(inline(" ".join(lines)), note_st)]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff6dd")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor("#e0a800")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    return t


def build(dst, sources):
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="ЛР3: подготовка к защите")
    lines = []
    for s in sources:
        lines += open(s, encoding="utf-8").read().splitlines() + [""]
    story, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            if story:
                story.append(PageBreak())
            story.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            story += [CondPageBreak(40 * mm), Paragraph(inline(line[3:]), h2)]
        elif line.startswith("### "):
            story += [CondPageBreak(25 * mm), Paragraph(inline(line[4:]), h3)]
        elif line.startswith("@@code"):
            _, path, rng = line.split()
            a, b = map(int, rng.split("-"))
            src = open(path, encoding="utf-8").read().splitlines()
            story += [code_block(src[a - 1:b], list(range(a, b + 1))), Spacer(1, 4)]
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story += [code_block(block), Spacer(1, 4)]
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            story += [make_table(rows, doc.width), Spacer(1, 5)]
            continue
        elif line.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i][1:].strip())
                i += 1
            story += [note(block, doc.width), Spacer(1, 5)]
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            story += [ListFlowable([ListItem(Paragraph(inline(t), base), leftIndent=12) for t in items],
                                   bulletType="bullet", start="•", leftIndent=12), Spacer(1, 3)]
            continue
        elif line.strip():
            story += [Paragraph(inline(line), base), Spacer(1, 3.5)]
        i += 1

    def footer(canvas, d):
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"стр. {d.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2:])
