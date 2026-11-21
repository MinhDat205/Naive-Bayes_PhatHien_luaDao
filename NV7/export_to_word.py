import os
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = r"D:\NV7"
MD_PATH = os.path.join(BASE_DIR, "Bao_Cao_Danh_Gia_Mo_Hinh.md")
DOCX_PATH = os.path.join(BASE_DIR, "Bao_Cao_Danh_Gia_Mo_Hinh.docx")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=90, bottom=90, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_bottom_border(paragraph, color_hex="1F4E79", size="12"):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:bottom w:val="single" w:sz="{size}" w:space="4" w:color="{color_hex}"/>'
        f"</w:pBdr>"
    )
    pPr.append(pBdr)


def clean_text(text):
    text = re.sub(r"[A-Za-z]:\\(?:[^\\\s`\"')]+\\)+", "", text)
    text = text.replace(r"$\rightarrow$", "→")
    text = text.replace(r"\rightarrow", "→")
    text = text.replace("<br>", "\n")
    text = text.replace(r"\|", "|")
    text = re.sub(r"\$\\mathbf\{([^}]+)\}\$", r"**\1**", text)
    text = re.sub(r"\\mathbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\in", "∈")
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = text.replace(r'""', '"')
    return text


def set_run_font(run, font_name="Times New Roman", size_pt=12, bold=False, italic=False, color=None):
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color


def add_formatted_runs(paragraph, text, base_font_size=12, base_color=None):
    text = clean_text(text)
    pattern = re.compile(r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`)")
    tokens = pattern.split(text)
    for token in tokens:
        if not token:
            continue
        run = paragraph.add_run()
        if token.startswith("**") and token.endswith("**"):
            inner = token[2:-2].replace("`", "")
            run.text = inner
            set_run_font(run, "Times New Roman", base_font_size, bold=True, color=base_color)
        elif token.startswith("*") and token.endswith("*"):
            inner = token[1:-1].replace("`", "")
            run.text = inner
            set_run_font(run, "Times New Roman", base_font_size, italic=True, color=base_color)
        elif token.startswith("`") and token.endswith("`"):
            inner = token[1:-1]
            run.text = inner
            set_run_font(
                run,
                "Times New Roman",
                base_font_size,
                bold=True,
                color=RGBColor(0x1F, 0x4E, 0x79) if base_color is None else base_color,
            )
        else:
            run.text = token
            set_run_font(run, "Times New Roman", base_font_size, color=base_color)


def split_md_table_row(line):
    # Split on unescaped pipe characters |
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    parts = re.split(r"(?<!\\)\|", stripped)
    return [p.strip() for p in parts]


def add_table_from_md(doc, table_lines):
    rows_data = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = split_md_table_row(stripped)
        if all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
            continue
        rows_data.append(cells)

    if not rows_data:
        return

    num_cols = max(len(r) for r in rows_data)
    table = doc.add_table(rows=len(rows_data), cols=num_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    col_widths = None
    if num_cols == 7:
        col_widths = [Cm(2.1), Cm(1.8), Cm(2.3), Cm(2.3), Cm(4.5), Cm(2.2), Cm(1.8)]
    elif num_cols == 6:
        col_widths = [Cm(1.0), Cm(2.2), Cm(2.1), Cm(2.1), Cm(5.8), Cm(3.8)]
    elif num_cols == 5:
        col_widths = [Cm(3.6), Cm(3.2), Cm(3.4), Cm(3.4), Cm(3.4)]
    elif num_cols == 3:
        col_widths = [Cm(5.0), Cm(5.5), Cm(6.3)]

    for r_idx, row_values in enumerate(rows_data):
        row = table.rows[r_idx]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if r_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for c_idx in range(num_cols):
            cell = row.cells[c_idx]
            if col_widths and c_idx < len(col_widths):
                cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=85, bottom=85, left=110, right=110)
            val = row_values[c_idx] if c_idx < len(row_values) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15

            if r_idx == 0:
                set_cell_background(cell, "1F4E79")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                clean_val = clean_text(val).replace("**", "").replace("`", "")
                run = p.add_run(clean_val)
                set_run_font(
                    run,
                    "Times New Roman",
                    10.5,
                    bold=True,
                    color=RGBColor(0xFF, 0xFF, 0xFF),
                )
            else:
                if r_idx % 2 == 0:
                    set_cell_background(cell, "F2F6FA")
                plain = clean_text(val).replace("**", "").replace("`", "")
                if num_cols == 6 and c_idx in (4, 5):
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                elif num_cols == 7 and c_idx == 4:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                elif num_cols == 3 and c_idx in (0, 2):
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                elif c_idx == 0 and num_cols != 6:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                elif len(plain) < 34 and not plain.startswith('"'):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                add_formatted_runs(p, val, base_font_size=10.5)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(2)
    spacer.paragraph_format.space_after = Pt(4)


def convert_md_to_docx(md_path, docx_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.0)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)

    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            add_table_from_md(doc, table_buffer)
            table_buffer = []

    img_pattern = re.compile(r"^!\[(.*?)\]\((.*?)\)$")

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            table_buffer.append(stripped)
            continue
        else:
            flush_table()

        if not stripped or stripped == "---":
            continue

        # Embedded Image ![caption](filename.png)
        img_match = img_pattern.match(stripped)
        if img_match:
            caption_text = clean_text(img_match.group(1).strip())
            img_file = img_match.group(2).strip()
            img_full_path = os.path.join(BASE_DIR, os.path.basename(img_file))
            if os.path.exists(img_full_path):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(6)
                p_img.paragraph_format.space_after = Pt(2)
                p_img.paragraph_format.keep_with_next = True
                run_img = p_img.add_run()
                run_img.add_picture(img_full_path, width=Cm(15.2))

                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(2)
                p_cap.paragraph_format.space_after = Pt(8)
                run_cap = p_cap.add_run(caption_text)
                set_run_font(
                    run_cap,
                    "Times New Roman",
                    11,
                    italic=True,
                    bold=True,
                    color=RGBColor(0x1F, 0x4E, 0x79),
                )
            continue

        # Main Title (# ...)
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(14)
            p.paragraph_format.line_spacing = 1.25
            title_text = clean_text(stripped[2:].strip()).replace("`", "")
            run = p.add_run(title_text)
            set_run_font(run, "Times New Roman", 15.5, bold=True, color=RGBColor(0x1F, 0x4E, 0x79))
            add_bottom_border(p, color_hex="1F4E79", size="14")
            continue

        # Major Heading (## ...)
        if stripped.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            heading_text = clean_text(stripped[3:].strip()).replace("`", "")
            run = p.add_run(heading_text)
            set_run_font(run, "Times New Roman", 14, bold=True, color=RGBColor(0x1F, 0x4E, 0x79))
            add_bottom_border(p, color_hex="1F4E79", size="8")
            continue

        # Sub-Heading (### ...)
        if stripped.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(11)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            sub_text = clean_text(stripped[4:].strip()).replace("`", "")
            run = p.add_run(sub_text)
            set_run_font(run, "Times New Roman", 13, bold=True, color=RGBColor(0x1F, 0x4E, 0x79))
            add_bottom_border(p, color_hex="B4C6E7", size="4")
            continue

        # Sub-Sub-Heading (#### ...)
        if stripped.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            sub4_text = clean_text(stripped[5:].strip()).replace("`", "")
            run = p.add_run(sub4_text)
            set_run_font(run, "Times New Roman", 12, bold=True, color=RGBColor(0x2E, 0x75, 0xB6))
            continue

        leading_spaces = len(line) - len(line.lstrip(" "))
        num_match = re.match(r"^(\d+\.)\s+(.*)$", stripped)
        if num_match:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.25
            run_num = p.add_run(num_match.group(1) + " ")
            set_run_font(run_num, "Times New Roman", 12, bold=True, color=RGBColor(0x1F, 0x4E, 0x79))
            add_formatted_runs(p, num_match.group(2), base_font_size=12)
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            level = 0 if leading_spaces < 2 else (1 if leading_spaces < 5 else 2)
            bullet_char = "• " if level == 0 else ("– " if level == 1 else "+ ")
            p.paragraph_format.left_indent = Cm(0.6 + level * 0.6)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.25
            run_b = p.add_run(bullet_char)
            set_run_font(
                run_b,
                "Times New Roman",
                12,
                bold=True,
                color=RGBColor(0x1F, 0x4E, 0x79) if level == 0 else None,
            )
            add_formatted_runs(p, stripped[2:].strip(), base_font_size=12)
            continue

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.25
        add_formatted_runs(p, stripped, base_font_size=12)

    flush_table()
    doc.save(docx_path)
    print(f"Đã xuất thành công báo cáo Word kèm bảng biểu và hình ảnh: {docx_path}")


if __name__ == "__main__":
    convert_md_to_docx(MD_PATH, DOCX_PATH)
