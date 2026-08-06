from __future__ import annotations

"""Export PDF Talys (ReportLab) — design moderne, charte institutionnelle."""

import re
import unicodedata
from pathlib import Path
from typing import Any

TALYS_ORANGE = (245, 130, 32)
TALYS_GREEN = (139, 197, 63)
TALYS_NAVY = (30, 64, 120)
TALYS_SLATE = (71, 85, 105)
TALYS_LIGHT = (248, 250, 252)
RISK_COLORS = {
    "FAIBLE": TALYS_GREEN,
    "MODERE": (245, 158, 11),
    "ELEVE": (239, 68, 68),
}
RISK_LABELS = {"FAIBLE": "Faible", "MODERE": "Modéré", "ELEVE": "Élevé"}


def _register_fonts() -> tuple[str, str]:
    """Regular + bold Unicode (Arial Windows / DejaVu Linux)."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pairs = (
            (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ),
            (
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            ),
        )
        for regular_path, bold_path in pairs:
            if regular_path.exists():
                pdfmetrics.registerFont(TTFont("TalysRegular", str(regular_path)))
                if bold_path.exists():
                    pdfmetrics.registerFont(TTFont("TalysBold", str(bold_path)))
                    return "TalysRegular", "TalysBold"
                return "TalysRegular", "TalysRegular"
    except Exception:
        pass
    return "Helvetica", "Helvetica-Bold"


def _text(text: str, *, unicode_font: str) -> str:
    if unicode_font == "Helvetica":
        norm = unicodedata.normalize("NFKD", str(text))
        return norm.encode("ascii", "ignore").decode("ascii", "ignore")
    return str(text)


def _md_to_rl(text: str) -> str:
    """Convertit **gras** markdown en balises ReportLab."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))


def _risk_label(level: str | None) -> str:
    return RISK_LABELS.get(str(level or "").upper(), str(level or "—"))


def _color_rgb(rgb: tuple[int, int, int], alpha: float = 1.0):
    import importlib

    colors = importlib.import_module("reportlab.lib.colors")
    return colors.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)


def export_structured_pdf(data: dict[str, Any], out_path: Path, *, logo_path: Path | None = None) -> None:
    import importlib

    body_font, bold_font = _register_fonts()
    pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    colors_mod = importlib.import_module("reportlab.lib.colors")
    styles_mod = importlib.import_module("reportlab.lib.styles")
    platypus = importlib.import_module("reportlab.platypus")
    units = importlib.import_module("reportlab.lib.units")

    A4 = pagesizes.A4
    cm = units.cm
    colors = colors_mod
    getSampleStyleSheet = styles_mod.getSampleStyleSheet
    paragraph = styles_mod.ParagraphStyle
    SimpleDocTemplate = platypus.SimpleDocTemplate
    Paragraph = platypus.Paragraph
    Spacer = platypus.Spacer
    Table = platypus.Table
    TableStyle = platypus.TableStyle
    Image = platypus.Image
    KeepTogether = platypus.KeepTogether

    page_w, page_h = A4
    margin_x = 2.0 * cm
    margin_top = 2.2 * cm
    margin_bottom = 2.0 * cm

    cin = str(data.get("cin", ""))
    profile = data.get("client_profile") or {}
    systems = data.get("systems") or {}
    worst = str(data.get("worst_risk", "—")).upper()
    risk_rgb = RISK_COLORS.get(worst, TALYS_SLATE)

    def on_page(canvas, doc):
        canvas.saveState()
        # Bandeau supérieur Talys
        canvas.setFillColor(_color_rgb(TALYS_ORANGE))
        canvas.rect(0, page_h - 0.45 * cm, page_w, 0.45 * cm, fill=1, stroke=0)
        canvas.setFillColor(_color_rgb(TALYS_GREEN))
        canvas.rect(0, page_h - 0.55 * cm, page_w * 0.35, 0.1 * cm, fill=1, stroke=0)

        # En-tête texte
        canvas.setFillColor(_color_rgb(TALYS_NAVY))
        canvas.setFont(bold_font, 8)
        canvas.drawString(margin_x, page_h - 1.15 * cm, "TALYS CONSULTING")
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(_color_rgb(TALYS_SLATE))
        canvas.drawString(margin_x, page_h - 1.45 * cm, f"Dossier crédit · CIN {cin}")

        # Pied de page
        canvas.setStrokeColor(_color_rgb(TALYS_ORANGE, 0.35))
        canvas.setLineWidth(0.5)
        canvas.line(margin_x, 1.35 * cm, page_w - margin_x, 1.35 * cm)
        canvas.setFont(body_font, 7)
        canvas.setFillColor(_color_rgb(TALYS_SLATE))
        canvas.drawString(margin_x, 0.85 * cm, "Document confidentiel — usage interne comité crédit")
        canvas.drawRightString(page_w - margin_x, 0.85 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
        title=f"Dossier crédit CIN {cin}",
        author=str(data.get("analyst", "Talys Consulting")),
    )

    styles = getSampleStyleSheet()
    title_style = paragraph(
        "TalysTitle",
        parent=styles["Heading1"],
        fontName=bold_font,
        fontSize=22,
        leading=28,
        textColor=_color_rgb(TALYS_NAVY),
        spaceAfter=6,
    )
    subtitle_style = paragraph(
        "TalysSubtitle",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=11,
        leading=15,
        textColor=_color_rgb(TALYS_SLATE),
        spaceAfter=14,
    )
    h2 = paragraph(
        "TalysH2",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12,
        leading=16,
        textColor=_color_rgb(TALYS_NAVY),
        spaceBefore=16,
        spaceAfter=8,
        borderPadding=(0, 0, 4, 0),
    )
    h3 = paragraph(
        "TalysH3",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=10,
        leading=14,
        textColor=_color_rgb(TALYS_ORANGE),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = paragraph(
        "TalysBody",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9.5,
        leading=14,
        textColor=_color_rgb((30, 41, 59)),
    )
    small = paragraph(
        "TalysSmall",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=7.5,
        leading=10,
        textColor=_color_rgb(TALYS_SLATE),
    )
    kpi_label = paragraph(
        "TalysKpiLabel",
        parent=body,
        fontName=body_font,
        fontSize=8,
        leading=10,
        textColor=_color_rgb(TALYS_SLATE),
    )
    kpi_value = paragraph(
        "TalysKpiValue",
        parent=body,
        fontName=bold_font,
        fontSize=13,
        leading=16,
        textColor=_color_rgb(TALYS_NAVY),
    )
    highlight = paragraph(
        "TalysHighlight",
        parent=body,
        fontName=body_font,
        fontSize=10,
        leading=15,
        textColor=_color_rgb((30, 41, 59)),
        leftIndent=6,
        rightIndent=6,
    )

    def P(text: str, style=body):
        safe = _md_to_rl(_text(text, unicode_font=body_font))
        return Paragraph(safe, style)

    story: list[Any] = []

    # ── En-tête dossier ──────────────────────────────────────────────────────
    header_row: list[Any] = []
    if logo_path and logo_path.exists():
        try:
            header_row.append(Image(str(logo_path), width=4.2 * cm, height=2.0 * cm))
        except Exception:
            pass
    client_name = ""
    if profile.get("nom"):
        client_name = f"{profile.get('prenom', '')} {profile.get('nom', '')}".strip()
    meta_lines = [
        f"<b>CIN</b> {cin}",
        f"<b>Client</b> {client_name or '—'}",
        f"<b>Ville</b> {profile.get('ville', '—')}",
        f"<b>Analyste</b> {data.get('analyst', '—')}",
        f"<b>Date</b> {data.get('generated_at', '—')}",
    ]
    meta_para = Paragraph("<br/>".join(_text(l, unicode_font=body_font) for l in meta_lines), body)
    if header_row:
        header_table = Table([[header_row[0], meta_para]], colWidths=[5 * cm, 11.5 * cm])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        story.append(header_table)
    else:
        story.append(meta_para)
    story.append(Spacer(1, 0.35 * cm))

    story.append(P("Dossier d'analyse crédit", title_style))
    story.append(P("Synthèse décisionnelle pour comité — modèles ML, systèmes institutionnels & RAG", subtitle_style))

    # Badge risque principal
    risk_badge = Table(
        [[
            P(f"Niveau de risque global · {_risk_label(worst)}", kpi_value),
            P(f"Probabilité défaut {float(data.get('worst_proba') or 0) * 100:.1f} %", kpi_label),
        ]],
        colWidths=[10 * cm, 6.5 * cm],
    )
    risk_badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(risk_rgb, 0.12)),
        ("BOX", (0, 0), (-1, -1), 1, _color_rgb(risk_rgb, 0.55)),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(risk_badge)
    story.append(Spacer(1, 0.45 * cm))

    # ── KPI cards (grille 3×2) ───────────────────────────────────────────────
    kpi_items: list[tuple[str, str]] = [
        ("Score KYC", str(data.get("kyc_score", "—"))),
        ("Probabilité défaut (max)", f"{float(data.get('worst_proba') or 0) * 100:.1f} %"),
        ("Convergence modèles", "Oui" if data.get("models_agree") else "Non"),
    ]
    if systems.get("institutional_score") is not None:
        kpi_items.extend([
            ("Score institutionnel", f"{float(systems['institutional_score']):.1%}"),
            ("Conformité réglementaire", f"{systems.get('rules', {}).get('compliance_score', '—')}/100"),
            ("Priorité watchlist", str(systems.get("early_warning", {}).get("watchlist_priority", "—"))),
        ])

    def _kpi_cell(label: str, value: str):
        inner = Table([[P(label, kpi_label)], [P(value, kpi_value)]], colWidths=[5.3 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(TALYS_LIGHT)),
            ("BOX", (0, 0), (-1, -1), 0.5, _color_rgb(TALYS_ORANGE, 0.25)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return inner

    kpi_rows: list[list[Any]] = []
    row_buf: list[Any] = []
    for label, value in kpi_items:
        row_buf.append(_kpi_cell(label, value))
        if len(row_buf) == 3:
            kpi_rows.append(row_buf)
            row_buf = []
    if row_buf:
        while len(row_buf) < 3:
            row_buf.append(Spacer(1, 1))
        kpi_rows.append(row_buf)
    if kpi_rows:
        kpi_grid = Table(kpi_rows, colWidths=[5.3 * cm, 5.3 * cm, 5.3 * cm], hAlign="LEFT")
        kpi_grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(kpi_grid)
        story.append(Spacer(1, 0.5 * cm))

    # ── Résumé exécutif (encadré) ────────────────────────────────────────────
    exec_text = data.get("executive_summary") or data.get("conclusion", "")
    exec_box = Table([[P(exec_text, highlight)]], colWidths=[16.5 * cm])
    exec_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(TALYS_ORANGE, 0.06)),
        ("LINELEFT", (0, 0), (0, -1), 3, _color_rgb(TALYS_ORANGE)),
        ("BOX", (0, 0), (-1, -1), 0.5, _color_rgb(TALYS_ORANGE, 0.2)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(KeepTogether([P("Résumé exécutif", h2), exec_box]))
    story.append(Spacer(1, 0.35 * cm))

    # ── Mots-clés ────────────────────────────────────────────────────────────
    keywords = data.get("keywords") or []
    if keywords:
        kw_text = "   ·   ".join(keywords)
        kw_box = Table([[P(kw_text, small)]], colWidths=[16.5 * cm])
        kw_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(TALYS_LIGHT)),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(P("Axes d'analyse", h2))
        story.append(kw_box)
        story.append(Spacer(1, 0.3 * cm))

    # ── Systèmes institutionnels ─────────────────────────────────────────────
    if systems:
        story.append(P("Systèmes décisionnels institutionnels", h2))
        rules = systems.get("rules") or {}
        ews = systems.get("early_warning") or {}
        inst_reco = systems.get("recommendation") or {}

        def _system_card(title: str, subtitle: str, body_text: str, accent: tuple[int, int, int]):
            card = Table([
                [P(title, h3)],
                [P(subtitle, kpi_label)],
                [P(body_text[:500], body)],
            ], colWidths=[16.5 * cm])
            card.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("LINELEFT", (0, 0), (0, -1), 2.5, _color_rgb(accent)),
                ("BOX", (0, 0), (-1, -1), 0.5, _color_rgb(accent, 0.2)),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            return card

        story.append(_system_card(
            "Business Rules Engine",
            f"Conformité {rules.get('compliance_score', '—')}/100 · "
            f"{rules.get('triggered_count', 0)} règle(s) déclenchée(s)",
            rules.get("summary") or "Analyse réglementaire non disponible.",
            TALYS_NAVY,
        ))
        story.append(Spacer(1, 0.2 * cm))
        story.append(_system_card(
            "Early Warning System",
            f"Priorité watchlist · {ews.get('watchlist_priority', 'NONE')}",
            ews.get("summary") or "Aucune alerte précoce significative.",
            TALYS_ORANGE,
        ))
        story.append(Spacer(1, 0.2 * cm))
        reco_actions = inst_reco.get("recommended_actions") or []
        reco_body = (inst_reco.get("justification") or "—")
        if reco_actions:
            reco_body += "\n\nActions recommandées :\n" + "\n".join(f"• {a}" for a in reco_actions[:5])
        story.append(_system_card(
            "Recommandation IA institutionnelle",
            f"Décision · {inst_reco.get('decision_label', '—')} "
            f"(confiance {float(inst_reco.get('confidence') or 0) * 100:.0f} %)"
            if inst_reco.get("confidence") is not None
            else f"Décision · {inst_reco.get('decision_label', '—')}",
            reco_body,
            TALYS_GREEN,
        ))
        story.append(Spacer(1, 0.35 * cm))

    # ── Recommandation métier ─────────────────────────────────────────────────
    rec = data.get("recommendation") or {}
    rec_rows = [
        [P("<b>Décision proposée</b>", body), P(rec.get("decision", "—"), body)],
        [P("<b>Actions opérationnelles</b>", body), P(rec.get("actions", "—"), body)],
        [P("<b>Plan de surveillance</b>", body), P(rec.get("surveillance", "—"), body)],
        [P("<b>Statut KYC</b>", body), P(rec.get("kyc_note", "—"), body)],
    ]
    rec_table = Table(rec_rows, colWidths=[5 * cm, 11.5 * cm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _color_rgb(TALYS_NAVY, 0.08)),
        ("TEXTCOLOR", (0, 0), (0, -1), _color_rgb(TALYS_NAVY)),
        ("FONTNAME", (0, 0), (0, -1), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.5, _color_rgb(TALYS_SLATE, 0.2)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether([P("Recommandation comité crédit", h2), rec_table]))
    story.append(Spacer(1, 0.35 * cm))

    # ── Modèles ML ───────────────────────────────────────────────────────────
    model_rows = [["Modèle analytique", "Risque", "Probabilité", "KYC", "Technologie"]]
    model_keys: list[tuple[str, str]] = [
        ("Deep Tabular", "classic"),
        ("Temporal Transformer", "sequential"),
        ("Graph Attention (GAT)", "graph"),
    ]
    model_risk_levels: list[str] = []
    for label, key in model_keys:
        obj = data.get(key) or {}
        model_risk_levels.append(str(obj.get("risk_level", "—")).upper())
        model_rows.append([
            P(label, body),
            P(_risk_label(obj.get("risk_level")), kpi_value),
            P(str(obj.get("default_proba", "—")), body),
            P(str(obj.get("kyc_score", "—")), body),
            P(str(obj.get("model_used", "—")), small),
        ])

    model_table = Table(model_rows, colWidths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 5 * cm])
    model_style = [
        ("BACKGROUND", (0, 0), (-1, 0), _color_rgb(TALYS_NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, _color_rgb(TALYS_SLATE, 0.15)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _color_rgb(TALYS_LIGHT)]),
    ]
    for i, lv in enumerate(model_risk_levels, start=1):
        rc = RISK_COLORS.get(lv, TALYS_SLATE)
        model_style.append(("BACKGROUND", (1, i), (1, i), _color_rgb(rc, 0.18)))
    model_table.setStyle(TableStyle(model_style))
    story.append(KeepTogether([P("Analyse multi-modèles (Machine Learning)", h2), model_table]))
    story.append(Spacer(1, 0.35 * cm))

    # ── Conclusion ───────────────────────────────────────────────────────────
    story.append(P("Conclusion & prochaines étapes", h2))
    story.append(P(data.get("conclusion", ""), body))
    story.append(Spacer(1, 0.25 * cm))
    story.append(P(
        "Prochaines étapes suggérées : validation en comité crédit, archivage du dossier, "
        "mise à jour du plan de surveillance et suivi des actions correctives identifiées.",
        small,
    ))

    # ── Références RAG ───────────────────────────────────────────────────────
    sources = data.get("sources") or []
    if sources:
        story.append(P("Références documentaires (RAG)", h2))
        for idx, s in enumerate(sources[:8], start=1):
            excerpt = str(s.get("text", ""))[:180].replace("\n", " ").strip()
            score_pct = float(s.get("score", 0)) * 100
            ref_box = Table([[
                P(
                    f"<b>[{idx}]</b> {s.get('source')} · fragment #{s.get('chunk_id')} "
                    f"· pertinence {score_pct:.0f} %<br/>{excerpt}…",
                    small,
                ),
            ]], colWidths=[16.5 * cm])
            ref_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(TALYS_LIGHT)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(ref_box)
            story.append(Spacer(1, 0.12 * cm))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    disclaimer = Table([[
        P(
            "<b>Avertissement</b> — Ce document est généré à des fins pédagogiques et d'aide à la décision. "
            "Les données peuvent être synthétiques. Il ne constitue ni un avis juridique, ni une décision "
            "automatique de crédit. La décision finale appartient à l'analyste crédit et au comité, "
            "conformément aux politiques internes Talys Consulting.",
            small,
        ),
    ]], colWidths=[16.5 * cm])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _color_rgb(TALYS_SLATE, 0.08)),
        ("BOX", (0, 0), (-1, -1), 0.5, _color_rgb(TALYS_SLATE, 0.25)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
