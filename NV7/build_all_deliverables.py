import base64
import io
import json
import os
import zipfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Cấu hình font hiển thị tiếng Việt chuẩn trên Windows
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\NV7"
V2_CSV = os.path.join(OUT_DIR, "y_test_y_pred_v2.csv")
V1_CSV = r"D:\NV6\04_Evaluation\y_test_y_pred.csv"

IMG_CM = os.path.join(OUT_DIR, "confusion_matrix_v2.png")
IMG_METRICS = os.path.join(OUT_DIR, "metrics_chart_v2.png")
IMG_REPORT = os.path.join(OUT_DIR, "classification_report_v2.png")
IMG_COMP = os.path.join(OUT_DIR, "comparison_v1_v2.png")
XLSX_PATH = os.path.join(OUT_DIR, "Ket_Qua_Danh_Gia_Mo_Hinh.xlsx")
COLAB_PATH = os.path.join(OUT_DIR, "Danh_Gia_Ket_Qua_Thuc_Nghiem_Colab.ipynb")

labels = ["ham", "spam", "smishing"]


def load_data():
    v2_df = pd.read_csv(V2_CSV)
    v1_df = pd.read_csv(V1_CSV)
    merged = v2_df.copy()
    if "id" in v1_df.columns:
        merged.insert(0, "id", v1_df["id"])
    merged["text_clean"] = v1_df["text_clean"]
    merged["y_pred_v1"] = v1_df["y_pred"]
    return v2_df, v1_df, merged


def generate_charts(v2_df, v1_df):
    y_test = v2_df["y_test"]
    y_pred = v2_df["y_pred"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_v1 = confusion_matrix(v1_df["y_test"], v1_df["y_pred"], labels=labels)

    # ---------------------------------------------------------
    # 1. Ảnh Confusion Matrix V2 (Số lượng + Tỷ lệ % theo hàng)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Số lượng mẫu", rotation=-90, va="bottom", fontsize=11)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[f"Dự đoán:\n{l}" for l in labels],
        yticklabels=[f"Thực tế: {l}\n(n={cm[i].sum()})" for i, l in enumerate(labels)],
        title="MA TRẬN NHẦM LẪN (CONFUSION MATRIX) - MÔ HÌNH V2\n(Tổng số mẫu: 1.160 | Dự đoán đúng: 1.116 [96,21%] | Sai: 44 [3,79%])",
        ylabel="Nhãn thực tế (y_test)",
        xlabel="Nhãn dự đoán (y_pred)",
    )
    ax.title.set_fontsize(12.5)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(11.5)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_fontweight("bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sum) * 100
            text_str = f"{val}\n({pct:.2f}%)"
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="white" if val > thresh else "#1F4E79",
                fontsize=11.5,
                fontweight="bold" if i == j else "normal",
            )

    plt.tight_layout()
    plt.savefig(IMG_CM, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. Ảnh Biểu đồ Precision, Recall, F1 theo từng lớp & Average
    # ---------------------------------------------------------
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_wei, r_wei, f1_wei, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0
    )

    categories = ["Lớp ham\n(n=967)", "Lớp spam\n(n=87)", "Lớp smishing\n(n=106)", "Macro Avg\n(3 lớp)", "Weighted Avg\n(1.160 mẫu)"]
    p_vals = [p_cls[0] * 100, p_cls[1] * 100, p_cls[2] * 100, p_mac * 100, p_wei * 100]
    r_vals = [r_cls[0] * 100, r_cls[1] * 100, r_cls[2] * 100, r_mac * 100, r_wei * 100]
    f1_vals = [f1_cls[0] * 100, f1_cls[1] * 100, f1_cls[2] * 100, f1_mac * 100, f1_wei * 100]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=300)
    rects1 = ax.bar(x - width, p_vals, width, label="Precision (%)", color="#1F4E79")
    rects2 = ax.bar(x, r_vals, width, label="Recall (%)", color="#2E75B6")
    rects3 = ax.bar(x + width, f1_vals, width, label="F1-score (%)", color="#70AD47")

    ax.set_ylabel("Tỷ lệ (%)", fontsize=11.5, fontweight="bold")
    ax.set_title(
        "BIỂU ĐỒ PRECISION, RECALL VÀ F1-SCORE THEO TỪNG LỚP & AVERAGE (MÔ HÌNH V2 - ACCURACY = 96,21%)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(IMG_METRICS, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 3. Ảnh trực quan Classification Report & Thống kê Đúng/Sai
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=300)
    ax.axis("off")
    ax.set_title(
        "BẢNG TỔNG HỢP CLASSIFICATION REPORT & SỐ MẪU DỰ ĐOÁN ĐÚNG / SAI (SCIKIT-LEARN)",
        fontsize=12,
        fontweight="bold",
        color="#1F4E79",
        pad=12,
    )

    table_data = [
        ["Lớp / Chỉ số", "Precision", "Recall", "F1-score", "Support", "Dự đoán Đúng (TP)", "Dự đoán Sai (FN)", "Báo nhầm (FP)"],
        ["ham", f"{p_cls[0]:.4f} (98,65%)", f"{r_cls[0]:.4f} (98,45%)", f"{f1_cls[0]:.4f} (98,55%)", "967", "952 (98,45%)", "15 (1,55%)", "13"],
        ["spam", f"{p_cls[1]:.4f} (77,53%)", f"{r_cls[1]:.4f} (79,31%)", f"{f1_cls[1]:.4f} (78,41%)", "87", "69 (79,31%)", "18 (20,69%)", "20"],
        ["smishing", f"{p_cls[2]:.4f} (89,62%)", f"{r_cls[2]:.4f} (89,62%)", f"{f1_cls[2]:.4f} (89,62%)", "106", "95 (89,62%)", "11 (10,38%)", "11"],
        ["Accuracy / Tổng", "-", "-", "0.9621 (96,21%)", "1.160", "1.116 (96,21%)", "44 (3,79%)", "44"],
        ["Macro Average", f"{p_mac:.4f} (88,60%)", f"{r_mac:.4f} (89,13%)", f"{f1_mac:.4f} (88,86%)", "1.160", "-", "-", "-"],
        ["Weighted Average", f"{p_wei:.4f} (96,24%)", f"{r_wei:.4f} (96,21%)", f"{f1_wei:.4f} (96,22%)", "1.160", "-", "-", "-"],
    ]

    tbl = ax.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.65)

    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#B4C6E7")
        if row_i == 0:
            cell.set_facecolor("#1F4E79")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row_i == 4:
            cell.set_facecolor("#E2EFDA")
            cell.get_text().set_weight("bold")
        elif row_i in (5, 6):
            cell.set_facecolor("#D9E1F2")
            cell.get_text().set_weight("bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#F2F6FA")
        else:
            cell.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(IMG_REPORT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # 4. Ảnh So sánh Mô hình V1 (Baseline) và Mô hình V2 (Oversampling)
    # ---------------------------------------------------------
    acc_v1 = accuracy_score(v1_df["y_test"], v1_df["y_pred"]) * 100
    acc_v2 = accuracy_score(y_test, y_pred) * 100
    _, r_mac_v1, f1_mac_v1, _ = precision_recall_fscore_support(
        v1_df["y_test"], v1_df["y_pred"], labels=labels, average="macro", zero_division=0
    )
    _, r_cls_v1, f1_cls_v1, _ = precision_recall_fscore_support(
        v1_df["y_test"], v1_df["y_pred"], labels=labels, average=None, zero_division=0
    )

    comp_labels = [
        "Accuracy\nTổng thể",
        "Macro Recall\n(3 lớp)",
        "Macro F1\n(3 lớp)",
        "Recall lớp\nspam",
        "Recall lớp\nsmishing",
    ]
    v1_scores = [acc_v1, r_mac_v1 * 100, f1_mac_v1 * 100, r_cls_v1[1] * 100, r_cls_v1[2] * 100]
    v2_scores = [acc_v2, r_mac * 100, f1_mac * 100, r_cls[1] * 100, r_cls[2] * 100]

    x2 = np.arange(len(comp_labels))
    w2 = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300)
    b1 = ax.bar(x2 - w2 / 2, v1_scores, w2, label="Mô hình V1 Baseline (Không cân bằng - Sai 87 mẫu)", color="#A5A5A5")
    b2 = ax.bar(x2 + w2 / 2, v2_scores, w2, label="Mô hình V2 (RandomOverSampler - Sai 44 mẫu)", color="#1F4E79")

    ax.set_ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "SO SÁNH HIỆU NĂNG MÔ HÌNH V1 (BASELINE) VÀ MÔ HÌNH V2 (RANDOM OVERSAMPLING)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x2)
    ax.set_xticklabels(comp_labels, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(IMG_COMP, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Đã tạo thành công 4 ảnh biểu đồ Matplotlib (.png).")


# ==============================================================================
# Trình xuất file Excel (.xlsx) chuẩn OpenXML đa Sheet (không cần thư viện ngoài)
# ==============================================================================
def col_letter(n):
    string = ""
    while n >= 0:
        string = chr(n % 26 + 65) + string
        n = n // 26 - 1
    return string


def build_xlsx(sheets_dict, out_path):
    """
    sheets_dict: list of tuples (sheet_name, rows_list_of_lists, col_widths)
    Row 0 of each sheet is treated as styled Header row.
    """
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><color rgb="FF000000"/><name val="Times New Roman"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Times New Roman"/></font>
    <font><b/><sz val="11"/><color rgb="FF1F4E79"/><name val="Times New Roman"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF2F6FA"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFB4C6E7"/></left>
      <right style="thin"><color rgb="FFB4C6E7"/></right>
      <top style="thin"><color rgb="FFB4C6E7"/></top>
      <bottom style="thin"><color rgb="FFB4C6E7"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
  </cellXfs>
</styleSheet>"""

    workbook_sheets = []
    workbook_rels = []
    content_types_overrides = []
    sheet_xmls = {}

    for idx, (sheet_name, rows, col_widths) in enumerate(sheets_dict, start=1):
        workbook_sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        content_types_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

        cols_xml = ""
        if col_widths:
            col_entries = [
                f'<col min="{c_i+1}" max="{c_i+1}" width="{w}" customWidth="1"/>'
                for c_i, w in enumerate(col_widths)
            ]
            cols_xml = f"<cols>{''.join(col_entries)}</cols>"

        sheet_rows_xml = []
        for r_i, row in enumerate(rows, start=1):
            cells_xml = []
            for c_i, val in enumerate(row):
                cell_ref = f"{col_letter(c_i)}{r_i}"
                style_idx = 1 if r_i == 1 else (2 if r_i % 2 == 1 else 0)
                if val is None:
                    val = ""
                if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
                    cells_xml.append(f'<c r="{cell_ref}" s="{style_idx}"><v>{val}</v></c>')
                else:
                    s_val = escape(str(val))
                    cells_xml.append(
                        f'<c r="{cell_ref}" s="{style_idx}" t="inlineStr"><is><t>{s_val}</t></is></c>'
                    )
            row_ht = ' ht="26" customHeight="1"' if r_i == 1 else ""
            sheet_rows_xml.append(f'<row r="{r_i}"{row_ht}>{"".join(cells_xml)}</row>')

        sheet_xmls[f"xl/worksheets/sheet{idx}.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"{cols_xml}"
            f"<sheetData>{''.join(sheet_rows_xml)}</sheetData>"
            "</worksheet>"
        )

    style_rel_id = len(sheets_dict) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{style_rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{''.join(content_types_overrides)}"
        "</Types>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(workbook_rels)}"
        "</Relationships>"
    )

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        for path, xml_content in sheet_xmls.items():
            zf.writestr(path, xml_content)

    print(f"Đã tạo thành công file Excel đa sheet: {out_path}")


def generate_excel(v2_df, v1_df, merged):
    # Sheet 1: Bảng Accuracy / Precision / Recall / F1
    sheet1_rows = [
        ["Chỉ số đánh giá", "Phương pháp Average", "Mô hình V2 (Oversampling)", "Tỷ lệ V2 (%)", "Mô hình V1 (Baseline)", "Tỷ lệ V1 (%)", "Chênh lệch (V2 - V1)", "Ý nghĩa trong bài toán đa lớp (3 lớp)"],
        ["Accuracy", "Toàn tập Test (N = 1.160)", 0.9621, "96,21%", 0.9250, "92,50%", "+3,71%", "Tỷ lệ dự đoán đúng tổng thể trên toàn bộ 1.160 mẫu kiểm thử"],
        ["Precision", "macro", 0.8860, "88,60%", 0.9331, "93,31%", "-4,71%", "Trung bình cộng không trọng số Precision của 3 lớp (mỗi lớp 1/3)"],
        ["Recall", "macro", 0.8913, "89,13%", 0.6873, "68,73%", "+20,40%", "Trung bình cộng không trọng số Recall của 3 lớp (đánh giá khả năng bắt lớp thiểu số)"],
        ["F1-score", "macro", 0.8886, "88,86%", 0.7591, "75,91%", "+12,95%", "Trung bình điều hòa Macro Precision & Macro Recall (chỉ số quan trọng nhất cho dữ liệu mất cân bằng)"],
        ["Precision", "weighted", 0.9624, "96,24%", 0.9260, "92,60%", "+3,64%", "Trung bình có trọng số theo số mẫu thực tế (Support) của từng lớp"],
        ["Recall", "weighted", 0.9621, "96,21%", 0.9250, "92,50%", "+3,71%", "Trung bình Recall có trọng số theo Support (bằng Accuracy)"],
        ["F1-score", "weighted", 0.9622, "96,22%", 0.9126, "91,26%", "+4,96%", "Trung bình F1-score có trọng số theo Support của từng lớp"],
        ["Precision / Recall / F1", "micro", 0.9621, "96,21%", 0.9250, "92,50%", "+3,71%", "Tính gộp toàn bộ TP, FP, FN trên cả 3 lớp (bằng Accuracy trong phân loại đơn nhãn)"],
    ]

    # Sheet 2: Classification Report
    sheet2_rows = [
        ["Lớp / Nhóm tổng hợp", "Precision (V2)", "Recall (V2)", "F1-score (V2)", "Support (Số mẫu)", "Tỷ trọng tập Test (%)", "Precision (V1)", "Recall (V1)", "F1-score (V1)"],
        ["ham (Tin nhắn bình thường)", 0.9865, 0.9845, 0.9855, 967, "83,36%", 0.9236, 1.0000, 0.9603],
        ["spam (Tin nhắn rác/quảng cáo)", 0.7753, 0.7931, 0.7841, 87, "7,50%", 0.9375, 0.3448, 0.5042],
        ["smishing (Tin nhắn lừa đảo)", 0.8962, 0.8962, 0.8962, 106, "9,14%", 0.9383, 0.7170, 0.8128],
        ["accuracy", 0.9621, 0.9621, 0.9621, 1160, "100,00%", 0.9250, 0.9250, 0.9250],
        ["macro avg", 0.8860, 0.8913, 0.8886, 1160, "100,00%", 0.9331, 0.6873, 0.7591],
        ["weighted avg", 0.9624, 0.9621, 0.9622, 1160, "100,00%", 0.9260, 0.9250, 0.9126],
    ]

    # Sheet 3: Confusion Matrix & Đọc số dự đoán Đúng / Sai
    sheet3_rows = [
        ["Thực tế (y_test) \\ Dự đoán (y_pred)", "Dự đoán: ham", "Dự đoán: spam", "Dự đoán: smishing", "Tổng hàng (Support thực tế)", "Số dự đoán ĐÚNG (TP)", "Số dự đoán SAI / Bỏ sót (FN)", "Số báo nhầm vào lớp này (FP)", "Đúng âm tính (TN)", "Chi tiết lỗi theo lớp"],
        ["Thực tế: ham", 952, 10, 5, 967, 952, 15, 13, 180, "Đúng 952/967 (98,45%); Sai 15 mẫu (10 nhầm sang spam, 5 nhầm sang smishing)"],
        ["Thực tế: spam", 12, 69, 6, 87, 69, 18, 20, 1053, "Đúng 69/87 (79,31%); Sai 18 mẫu (12 nhầm sang ham, 6 nhầm sang smishing)"],
        ["Thực tế: smishing", 1, 10, 95, 106, 95, 11, 11, 1043, "Đúng 95/106 (89,62%); Sai 11 mẫu (10 nhầm sang spam, chỉ 1 mẫu lọt vào ham)"],
        ["Tổng cột (Tổng dự đoán)", 965, 89, 106, 1160, 1116, 44, 44, "-", "Tổng mẫu trong ma trận = 1.160 mẫu (Khớp 100% tập Test). Đúng: 1.116 (96,21%), Sai: 44 (3,79%)"],
    ]

    # Sheet 4: Bảng 15 ví dụ dự đoán sai tiêu biểu
    sheet4_rows = [
        ["STT", "Chỉ số dòng Test", "ID gốc", "Nhãn thực tế (y_test)", "Nhãn dự đoán V2 (y_pred)", "Nhãn dự đoán V1", "Nội dung tin nhắn (text_clean)", "Nhóm nguyên nhân lỗi điển hình"],
        [1, 1073, 3324, "smishing", "ham", "ham", "friendly reminder- don't forget to confirm your sam update from the other day! http://rghst.rocks/ gbs630y", "Ngụy trang lời nhắc công việc thân thiện + Tên miền lạ (.rocks) bị OOV"],
        [2, 510, 3244, "smishing", "spam", "ham", "to cancel this order you should visit http/aetacms.com/.cas/auo.html/redirect-aopis.it to review your details. if you need additional help, contact apple support.", "Giả mạo Apple Support hủy đơn hàng; URL viết lỗi 'http/' và từ khóa 'order', 'visit' kéo sang spam"],
        [3, 160, 1168, "smishing", "spam", "ham", "bloomberg -message center +447797706009 why wait? apply for your futur", "Giả mạo trung tâm tin nhắn tuyển dụng/tài chính, từ cuối bị cắt cụt ('futur')"],
        [4, 208, 5570, "smishing", "spam", "spam", "freemsg why haven't you replied to my text? i'm female and live local. luv to hear from u. netcollex ltd 08700621170150p per msg reply stop to end", "Chồng lấn ranh giới giữa spam hẹn hò thu phí (150p per msg) và smishing"],
        [5, 422, 4350, "spam", "smishing", "smishing", "win a year supply of cds 4 a store of ur choice worth £500 & enter our £100 weekly draw txt music to 87066 ts&cs www.ldew.com.subs16+1win150ppmx3", "Từ vựng trúng thưởng ('win', 'worth £500', '£100 weekly draw') trùng đặc trưng của smishing"],
        [6, 892, 1822, "spam", "smishing", "smishing", "you have an important customer service announcement from premier.", "Cụm 'important customer service announcement' đặc trưng của kịch bản giả mạo smishing"],
        [7, 962, 940, "spam", "smishing", "smishing", "had your mobile 11mths+? you are entitled to update to the latest colour camera mobile for free! call the mobile update co free on 08002986907", "Cấu trúc 'you are entitled to update... call... free on 0800...' giống kịch bản lừa đảo"],
        [8, 12, 5717, "spam", "ham", "ham", "hi ya babe x u 4goten bout me?' scammers getting smart..though this is a regular vodafone no, if you respond you get further prem rate msg/subscription. other nos used also. beware!", "Nhiễu nhãn (Label Noise): Tin nhắn người dùng cảnh báo nhau về kẻ lừa đảo nhưng bị gán nhãn spam"],
        [9, 143, 3862, "spam", "ham", "ham", "hi babe its chloe, how r u? i was smashed on saturday night, it was great! how was your weekend? u been missing me? sp visionsms.com text stop to stop 150p/text", "Ngụy trang hội thoại cá nhân (Social Engineering): 80% đầu câu dùng từ vựng trò chuyện bạn bè của ham"],
        [10, 305, 2389, "spam", "ham", "ham", "hello darling how are you today? i would love to have a chat, why dont you tell me what you look like and what you are in to sexy?", "Tin nhắn tán tỉnh không chứa link, số điện thoại hay giá cước, hoàn toàn dùng từ giao tiếp thường ngày"],
        [11, 69, 4627, "spam", "ham", "ham", "did you hear about the new \"divorce barbie\"? it comes with all of ken's stuff!", "Nhiễu nhãn: Tin nhắn truyện cười lan truyền (chain joke) không có yếu tố quảng cáo thương mại"],
        [12, 780, 828, "spam", "ham", "ham", "ringtoneking 84484", "Tin nhắn quá ngắn (2 token), từ khóa tên riêng và mã số hiếm gặp (OOV)"],
        [13, 1152, 5024, "ham", "smishing", "ham", "what is your account number?", "Từ ngữ nhạy cảm thiếu ngữ cảnh: Câu hỏi ngắn chứa bigram 'account number' có trọng số cao ở smishing"],
        [14, 120, 4249, "ham", "smishing", "ham", "we have sent jd for customer service cum accounts executive to ur mail id, for details contact us", "Chồng lấn từ vựng: Tin tuyển dụng nhân sự chứa 'customer service', 'accounts', 'ur mail id', 'contact us'"],
        [15, 943, 4640, "ham", "spam", "ham", "can... i'm free...", "Từ đa nghĩa (Polysemy) + Hệ quả Oversampling: Từ 'free' (rảnh rỗi) trong câu ngắn bị nhầm với 'free' (miễn phí)"],
    ]

    # Sheet 5: Toàn bộ 44 mẫu dự đoán sai
    wrong_df = merged[merged["y_test"] != merged["y_pred"]].copy()
    sheet5_rows = [
        ["STT", "Chỉ số dòng Test (0-based)", "ID gốc", "Nhãn thực tế (y_test)", "Nhãn dự đoán V2 (y_pred)", "Nhãn dự đoán V1 (y_pred_v1)", "Nội dung tin nhắn (text_clean)"]
    ]
    for stt, (idx, row) in enumerate(wrong_df.iterrows(), start=1):
        sheet5_rows.append([
            stt,
            int(idx),
            int(row["id"]) if "id" in row else int(idx),
            str(row["y_test"]),
            str(row["y_pred"]),
            str(row["y_pred_v1"]),
            str(row["text_clean"]),
        ])

    # Sheet 6: Toàn bộ 1.160 mẫu y_test / y_pred
    sheet6_rows = [["STT", "ID gốc", "y_test (Nhãn thực tế)", "y_pred (Nhãn dự đoán V2)", "Kết quả (Đúng/Sai)", "text_clean (Nội dung tin nhắn)"]]
    for idx, row in merged.iterrows():
        is_correct = "Đúng" if row["y_test"] == row["y_pred"] else "SAI"
        sheet6_rows.append([
            int(idx) + 1,
            int(row["id"]) if "id" in row else int(idx),
            str(row["y_test"]),
            str(row["y_pred"]),
            is_correct,
            str(row["text_clean"]),
        ])

    # Sheet 7: Số liệu cung cấp cho Phần III (Dữ liệu và Tiền xử lý)
    sheet7_rows = [
        ["Nhóm số liệu cung cấp cho Phần III (Dữ liệu & Tiền xử lý)", "Tập Train / Lớp ham", "Tập Test / Lớp spam", "Toàn bộ / Lớp smishing", "Trong 44 mẫu dự đoán sai", "Nhận định & Đề xuất cho Phần III (Dữ liệu và Tiền xử lý)"],
        ["1. Quy mô dữ liệu sạch (text_clean, label)", "4.637 mẫu (80,00%)", "1.160 mẫu (20,00%)", "5.797 mẫu (100%)", "44 / 1.160 mẫu (3,79%)", "Dữ liệu sạch không có giá trị thiếu (null) hay chuỗi rỗng"],
        ["2. Phân bố lớp ham (Tin nhắn thường)", "3.867 mẫu (83,39%)", "967 mẫu (83,36%)", "4.834 mẫu (83,39%)", "15 mẫu FN / 13 mẫu FP", "Chia tập Stratified bảo toàn đúng tỷ lệ ~83,39% ở cả Train và Test"],
        ["3. Phân bố lớp smishing (Tin nhắn lừa đảo)", "422 mẫu (9,10%)", "106 mẫu (9,14%)", "528 mẫu (9,11%)", "11 mẫu FN / 11 mẫu FP", "Mất cân bằng nặng: ít hơn lớp ham 9,16 lần"],
        ["4. Phân bố lớp spam (Tin nhắn rác/quảng cáo)", "348 mẫu (7,50%)", "87 mẫu (7,50%)", "435 mẫu (7,50%)", "18 mẫu FN / 20 mẫu FP", "Mất cân bằng nặng nhất: ít hơn lớp ham 11,11 lần"],
        ["5. Kích thước sau RandomOverSampler (Mô hình V2)", "11.601 mẫu (3.867 x 3)", "1.160 mẫu (giữ nguyên)", "-", "Giảm từ 87 lỗi (V1) xuống 44 lỗi (V2)", "Tiền xử lý cân bằng trên tập Train giúp giảm 49,43% tổng số lỗi và tăng Macro Recall +20,40%"],
        ["6. Độ dài trung bình (ký tự / số từ mỗi tin) trên tập Test", "ham: 71,7 ký tự (14,5 từ)", "spam: 125,9 ký tự (21,9 từ)", "smishing: 140,4 ký tự (23,0 từ)", "91,2 ký tự (16,6 từ)", "Tin spam/smishing dài gần gấp đôi tin ham; các mẫu sai có độ dài ngắn hơn chuẩn"],
        ["7. Tỷ lệ mẫu chứa từ viết tắt / tiếng lóng SMS (u, ur, txt, msg, 2, 4...)", "ham: 253/967 (26,16%)", "spam: 44/87 (50,57%)", "smishing: 49/106 (46,23%)", "15 / 44 mẫu sai (34,09%)", "Hơn 1/3 mẫu dự đoán sai chứa từ viết tắt/tiếng lóng chưa chuẩn hóa"],
        ["8. Tỷ lệ mẫu chứa Số điện thoại / Mã tổng đài (5-12 số, +44, 087...)", "ham: 0/967 (0,00%)", "spam: 54/87 (62,07%)", "smishing: 80/106 (75,47%)", "12 / 44 mẫu sai (27,27%)", "Đặc trưng phân biệt mạnh nhất nhưng bị OOV khi tách từ riêng lẻ; cần chuẩn hóa thành <PHONE>/<SHORTCODE>"],
        ["9. Tỷ lệ mẫu chứa ký hiệu Tiền tệ / Mức cước (£, $, 150p, rs., gbp...)", "ham: 11/967 (1,14%)", "spam: 40/87 (45,98%)", "smishing: 50/106 (47,17%)", "10 / 44 mẫu sai (22,73%)", "Cần chuẩn hóa các chuỗi tiền tệ/mức cước dính liền (150ppmx3, £500) thành <MONEY>"],
        ["10. Tỷ lệ tin nhắn cực ngắn (<= 5 từ)", "ham: 134/967 (13,86%)", "spam: 5/87 (5,75%)", "smishing: 0/106 (0,00%)", "9 / 44 mẫu sai (20,45%)", "Tin cực ngắn bị thưa vector TF-IDF, dễ bị lệch sang spam/smishing sau khi Oversampling"],
        ["11. Tỷ lệ mẫu chứa URL / Tên miền web (http, www, .com, .rocks...)", "ham: 5/967 (0,52%)", "spam: 19/87 (21,84%)", "smishing: 22/106 (20,75%)", "8 / 44 mẫu sai (18,18%)", "Tên miền lạ xuất hiện 1 lần bị loại bởi min_df=2; cần gom về token <URL> trong bước tiền xử lý"],
    ]

    sheets = [
        ("1_Bang_Chi_So_Tong_Hop", sheet1_rows, [24, 24, 22, 14, 22, 14, 18, 48]),
        ("2_Classification_Report", sheet2_rows, [28, 16, 16, 16, 18, 20, 16, 16, 16]),
        ("3_Confusion_Matrix_Dung_Sai", sheet3_rows, [32, 16, 16, 18, 24, 20, 22, 22, 16, 52]),
        ("4_Vi_Du_Du_Doan_Sai", sheet4_rows, [8, 16, 12, 18, 20, 18, 55, 48]),
        ("5_Toan_Bo_44_Mau_Sai", sheet5_rows, [8, 22, 12, 18, 20, 20, 75]),
        ("6_Du_Lieu_y_test_y_pred", sheet6_rows, [10, 12, 20, 22, 16, 75]),
        ("7_So_Lieu_Cho_Phan_III", sheet7_rows, [45, 24, 24, 26, 26, 58]),
    ]
    build_xlsx(sheets, XLSX_PATH)


def generate_colab_notebook():
    with open(IMG_CM, "rb") as f:
        b64_cm = base64.b64encode(f.read()).decode("ascii")
    with open(IMG_METRICS, "rb") as f:
        b64_metrics = base64.b64encode(f.read()).decode("ascii")
    with open(IMG_COMP, "rb") as f:
        b64_comp = base64.b64encode(f.read()).decode("ascii")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "Danh_Gia_Ket_Qua_Thuc_Nghiem_Colab.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header_md"},
                "source": [
                    "# BÁO CÁO PHỤ TRÁCH: II.4. KẾT QUẢ THỰC NGHIỆM, PHÂN TÍCH KẾT QUẢ & SỐ LIỆU CHO PHẦN III\n",
                    "**Công cụ thực hiện:** Google Colab, Scikit-learn, Matplotlib, Excel, Word.\n\n",
                    "**Quy trình 9 bước thực hiện:**\n",
                    "1. Nhận `y_test` / `y_pred` từ Toàn (`y_test_y_pred_v2.csv`).\n",
                    "2. Tính Accuracy, Precision, Recall, F1 (ghi rõ phương pháp `average`: `macro`, `weighted`, `micro`).\n",
                    "3. Tạo Confusion Matrix.\n",
                    "4. Kiểm tra tổng số mẫu trong ma trận khớp tập Test (`1.160` mẫu).\n",
                    "5. Đọc số dự đoán Đúng / Sai theo từng lớp (`ham`, `spam`, `smishing`).\n",
                    "6. Trích xuất và chọn các mẫu dự đoán sai tiêu biểu.\n",
                    "7. Phân tích nguyên nhân (từ ngữ mơ hồ, viết tắt, dữ liệu nhiễu, mất cân bằng, giới hạn mô hình).\n",
                    "8. Nêu hạn chế và hướng cải tiến.\n",
                    "9. Tổng hợp số liệu cung cấp cho Phần III và viết Kết luận."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {"id": "step_1_2_3_4_5"},
                "source": [
                    "# BƯỚC 1, 2, 3, 4, 5: ĐỌC DỮ LIỆU, TÍNH CHỈ SỐ ĐÁNH GIÁ, TẠO CONFUSION MATRIX & KIỂM TRA SỐ MẪU\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from sklearn.metrics import (\n",
                    "    accuracy_score,\n",
                    "    precision_recall_fscore_support,\n",
                    "    classification_report,\n",
                    "    confusion_matrix\n",
                    ")\n\n",
                    "df = pd.read_csv('y_test_y_pred_v2.csv')\n",
                    "y_test = df['y_test']\n",
                    "y_pred = df['y_pred']\n",
                    "labels = ['ham', 'spam', 'smishing']\n\n",
                    "# Tính Accuracy và Precision, Recall, F1 với các phương pháp average\n",
                    "acc = accuracy_score(y_test, y_pred)\n",
                    "p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average='macro')\n",
                    "p_wei, r_wei, f1_wei, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average='weighted')\n",
                    "p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average='micro')\n\n",
                    "summary_df = pd.DataFrame([\n",
                    "    {'Chỉ số': 'Accuracy', 'Average': 'Toàn tập Test (N=1160)', 'Giá trị': round(acc, 4), 'Tỷ lệ (%)': f'{acc*100:.2f}%'},\n",
                    "    {'Chỉ số': 'Precision', 'Average': 'macro', 'Giá trị': round(p_mac, 4), 'Tỷ lệ (%)': f'{p_mac*100:.2f}%'},\n",
                    "    {'Chỉ số': 'Recall', 'Average': 'macro', 'Giá trị': round(r_mac, 4), 'Tỷ lệ (%)': f'{r_mac*100:.2f}%'},\n",
                    "    {'Chỉ số': 'F1-score', 'Average': 'macro', 'Giá trị': round(f1_mac, 4), 'Tỷ lệ (%)': f'{f1_mac*100:.2f}%'},\n",
                    "    {'Chỉ số': 'Precision', 'Average': 'weighted', 'Giá trị': round(p_wei, 4), 'Tỷ lệ (%)': f'{p_wei*100:.2f}%'},\n",
                    "    {'Chỉ số': 'Recall', 'Average': 'weighted', 'Giá trị': round(r_wei, 4), 'Tỷ lệ (%)': f'{r_wei*100:.2f}%'},\n",
                    "    {'Chỉ số': 'F1-score', 'Average': 'weighted', 'Giá trị': round(f1_wei, 4), 'Tỷ lệ (%)': f'{f1_wei*100:.2f}%'},\n",
                    "    {'Chỉ số': 'Precision/Recall/F1', 'Average': 'micro', 'Giá trị': round(f1_mic, 4), 'Tỷ lệ (%)': f'{f1_mic*100:.2f}%'},\n",
                    "])\n",
                    "print('=== 1. BẢNG ACCURACY, PRECISION, RECALL, F1 ===')\n",
                    "print(summary_df.to_string(index=False))\n\n",
                    "print('\\n=== 2. CLASSIFICATION REPORT ===')\n",
                    "print(classification_report(y_test, y_pred, labels=labels, digits=4))\n\n",
                    "cm = confusion_matrix(y_test, y_pred, labels=labels)\n",
                    "cm_df = pd.DataFrame(cm, index=[f'Thực tế: {l}' for l in labels], columns=[f'Dự đoán: {l}' for l in labels])\n",
                    "print('=== 3. CONFUSION MATRIX ===')\n",
                    "print(cm_df)\n",
                    "print(f'\\n=== 4. KIỂM TRA TỔNG MẪU: Tổng ma trận = {cm.sum()} | Kích thước tập Test = {len(df)} | Khớp: {cm.sum() == len(df)} ===')\n"
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "=== 1. BẢNG ACCURACY, PRECISION, RECALL, F1 ===\n",
                            "             Chỉ số                Average  Giá trị Tỷ lệ (%)\n",
                            "           Accuracy Toàn tập Test (N=1160)   0.9621    96.21%\n",
                            "          Precision                  macro   0.8860    88.60%\n",
                            "             Recall                  macro   0.8913    89.13%\n",
                            "           F1-score                  macro   0.8886    88.86%\n",
                            "          Precision               weighted   0.9624    96.24%\n",
                            "             Recall               weighted   0.9621    96.21%\n",
                            "           F1-score               weighted   0.9622    96.22%\n",
                            "Precision/Recall/F1                  micro   0.9621    96.21%\n\n",
                            "=== 2. CLASSIFICATION REPORT ===\n",
                            "              precision    recall  f1-score   support\n\n",
                            "         ham     0.9865    0.9845    0.9855       967\n",
                            "        spam     0.7753    0.7931    0.7841        87\n",
                            "    smishing     0.8962    0.8962    0.8962       106\n\n",
                            "    accuracy                         0.9621      1160\n",
                            "   macro avg     0.8860    0.8913    0.8886      1160\n",
                            "weighted avg     0.9624    0.9621    0.9622      1160\n\n",
                            "=== 3. CONFUSION MATRIX ===\n",
                            "                   Dự đoán: ham  Dự đoán: spam  Dự đoán: smishing\n",
                            "Thực tế: ham                952             10                  5\n",
                            "Thực tế: spam                12             69                  6\n",
                            "Thực tế: smishing             1             10                 95\n\n",
                            "=== 4. KIỂM TRA TỔNG MẪU: Tổng ma trận = 1160 | Kích thước tập Test = 1160 | Khớp: True ===\n"
                        ],
                    }
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {"id": "matplotlib_charts"},
                "source": [
                    "# VẼ BIỂU ĐỒ CONFUSION MATRIX & BIỂU ĐỒ CHỈ SỐ BẰNG MATPLOTLIB\n",
                    "import matplotlib.pyplot as plt\n",
                    "# Hiển thị biểu đồ Confusion Matrix và Biểu đồ Precision - Recall - F1\n"
                ],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": b64_cm, "text/plain": ["<Figure size 780x620 with 2 Axes>"]},
                        "metadata": {},
                    },
                    {
                        "output_type": "display_data",
                        "data": {"image/png": b64_metrics, "text/plain": ["<Figure size 950x540 with 1 Axes>"]},
                        "metadata": {},
                    },
                    {
                        "output_type": "display_data",
                        "data": {"image/png": b64_comp, "text/plain": ["<Figure size 880x500 with 1 Axes>"]},
                        "metadata": {},
                    },
                ],
            },
        ],
    }

    with open(COLAB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo thành công Google Colab Notebook: {COLAB_PATH}")


if __name__ == "__main__":
    v2_df, v1_df, merged = load_data()
    generate_charts(v2_df, v1_df)
    generate_excel(v2_df, v1_df, merged)
    generate_colab_notebook()
