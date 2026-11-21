"""Generate a polished French PDF report for the Telco Churn MLOps project."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path("output/pdf/rapport_mlops_telco_churn.pdf")
NAVY = colors.HexColor("#0F2747")
BLUE = colors.HexColor("#1769AA")
TEAL = colors.HexColor("#00897B")
ORANGE = colors.HexColor("#E67E22")
RED = colors.HexColor("#C0392B")
LIGHT = colors.HexColor("#F4F7FA")
MUTED = colors.HexColor("#5B6770")


class PipelineDiagram(Flowable):
    """Compact process diagram drawn as vectors for crisp PDF output."""

    def __init__(self) -> None:
        super().__init__()
        self.width = 17.0 * cm
        self.height = 8.6 * cm

    def draw_box(self, canvas: object, x: float, y: float, width: float, height: float, title: str, text: str, color: colors.Color) -> None:
        """Draw one labelled process box."""
        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.3)
        canvas.roundRect(x, y, width, height, 7, fill=1, stroke=1)
        canvas.setFillColor(color)
        canvas.setFont("Helvetica-Bold", 8.2)
        canvas.drawCentredString(x + width / 2, y + height - 13, title)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6.9)
        for index, line in enumerate(text.split("\n")):
            canvas.drawCentredString(x + width / 2, y + height - 26 - index * 8, line)

    def arrow(self, canvas: object, x1: float, y1: float, x2: float, y2: float, color: colors.Color = NAVY) -> None:
        """Draw a directional arrow."""
        canvas.setStrokeColor(color)
        canvas.setFillColor(color)
        canvas.setLineWidth(1.1)
        canvas.line(x1, y1, x2, y2)
        canvas.circle(x2, y2, 2.1, fill=1, stroke=0)

    def draw(self) -> None:
        """Render pipeline, decision branches, serving, and monitoring."""
        c = self.canv
        w, h = self.width, self.height
        c.setFillColor(LIGHT)
        c.roundRect(0, 0, w, h, 10, fill=1, stroke=0)
        y_top, box_w, box_h = h - 48, 78, 39
        x_positions = [10, 102, 194, 286, 378]
        labels = [
            ("LOT CSV", "Donnees clients\nbrutes", BLUE),
            ("VALIDATION", "Schema + qualite\ndes valeurs", ORANGE),
            ("ENTRAINEMENT", "Preparation\n+ Random Forest", TEAL),
            ("COMPARAISON", "Nouveau modele\nvs champion", NAVY),
            ("MLFLOW", "Versions, metriques\net artefacts", BLUE),
        ]
        for x, (title, text, color) in zip(x_positions, labels):
            self.draw_box(c, x, y_top, box_w, box_h, title, text, color)
        for x in x_positions[:-1]:
            self.arrow(c, x + box_w, y_top + box_h / 2, x + 90, y_top + box_h / 2)
        c.setFillColor(RED)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(105, y_top - 18, "Schema invalide -> arret et alerte")
        self.arrow(c, 139, y_top, 139, y_top - 38, RED)
        self.draw_box(c, 91, y_top - 85, 96, 35, "ALERTE", "Data scientist\nintervient", RED)
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(278, y_top - 18, "Meilleur modele -> promotion champion")
        self.arrow(c, 325, y_top, 325, y_top - 38, TEAL)
        self.draw_box(c, 277, y_top - 85, 96, 35, "API FASTAPI", "Prediction REST\nmodele champion", TEAL)
        self.arrow(c, 373, y_top - 67, 430, y_top - 67, NAVY)
        self.draw_box(c, 436, y_top - 85, 82, 35, "MONITORING", "Prometheus\n+ Grafana", NAVY)


def styles() -> dict[str, ParagraphStyle]:
    """Return consistently styled report text blocks."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=16, textColor=MUTED, alignment=TA_CENTER),
        "h1": ParagraphStyle("Heading", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=10, spaceAfter=8),
        "h2": ParagraphStyle("Subheading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4, leading=14, textColor=colors.HexColor("#263238"), spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED),
        "center": ParagraphStyle("Center", parent=base["BodyText"], alignment=TA_CENTER, fontName="Helvetica", fontSize=9.4, leading=13, textColor=colors.HexColor("#263238")),
    }


def footer(canvas: object, doc: object) -> None:
    """Add page number and project label."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EA"))
    canvas.line(1.8 * cm, 1.35 * cm, A4[0] - 1.8 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 0.92 * cm, "MLOps - Telco Churn | Rapport de demonstration")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.92 * cm, f"Page {doc.page}")
    canvas.restoreState()


def p(text: str, style: ParagraphStyle) -> Paragraph:
    """Create a safe report paragraph."""
    return Paragraph(text, style)


def make_table(rows: list[list[str]], widths: list[float]) -> Table:
    """Build a polished table with an emphasized header row."""
    header = ParagraphStyle("TableHeader", fontName="Helvetica-Bold", fontSize=7.6, leading=9.2, textColor=colors.white)
    cell = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=7.45, leading=9.3, textColor=colors.HexColor("#263238"))
    prepared = [[Paragraph(value, header if row_index == 0 else cell) for value in row] for row_index, row in enumerate(rows)]
    table = Table(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build() -> Path:
    """Build the final multi-page PDF report."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style = styles()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.65 * cm, bottomMargin=1.7 * cm, title="Rapport MLOps - Telco Churn", author="Projet MLOps")
    story: list[object] = []

    story += [Spacer(1, 2.3 * cm), p("RAPPORT DE PROJET", style["subtitle"]), Spacer(1, 0.25 * cm), p("Pipeline MLOps - Telco Churn", style["title"]), p("Industrialisation d'un modele de prediction du churn client", style["subtitle"]), Spacer(1, 1.0 * cm)]
    summary = [[p("Objectif", style["small"]), p("Construire un systeme complet : verifier les donnees, entrainer et comparer les modeles, exposer une prediction REST et surveiller le service.", style["body"])], [p("Resultat", style["small"]), p("Les quatre services Docker sont operationnels. Les scenarios normal, erreur de schema et derive ont ete executes avec les statuts attendus.", style["body"])]]
    summary_table = Table(summary, colWidths=[3.0 * cm, 13.0 * cm])
    summary_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2EA")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [summary_table, Spacer(1, 1.15 * cm), p("Architecture fonctionnelle", style["h1"]), p("Le schema suivant resume le parcours d'un lot de donnees, depuis sa validation jusqu'a sa surveillance en production.", style["body"]), PipelineDiagram(), Spacer(1, 0.3 * cm), p("Figure 1 - Vue synthetique du pipeline MLOps", style["center"]), PageBreak()]

    story += [p("1. Probleme et objectif", style["h1"]), p("Le projet predit le churn, c'est-a-dire le risque qu'un client telecom quitte son operateur. L'objectif n'est pas seulement de produire une prediction, mais de rendre le modele fiable et exploitable dans le temps.", style["body"]), p("Le pipeline traite un fichier CSV de clients. Il controle d'abord sa structure et ses valeurs. Selon le resultat, il arrete le traitement, ne fait rien, ou entraine un nouveau modele.", style["body"]), p("2. Composants mis en place", style["h1"])]
    components = [
        ["Composant", "Role simple", "Adresse locale"],
        ["MLflow", "Conserve les executions, les versions de modeles, les metriques et les artefacts.", "http://localhost:5000"],
        ["API FastAPI", "Recoit les informations d'un client et retourne une probabilite de churn.", "http://localhost:8000"],
        ["Prometheus", "Collecte regulierement les chiffres de fonctionnement de l'API.", "http://localhost:9090"],
        ["Grafana", "Transforme les chiffres collectes en graphiques et indicateurs faciles a lire.", "http://localhost:3000"],
    ]
    story += [make_table(components, [3.1 * cm, 8.2 * cm, 5.0 * cm]), Spacer(1, 0.45 * cm), p("3. Fonctionnement du pipeline", style["h1"])]
    steps = [
        ("Validation", "Le fichier est controle : colonnes attendues, valeurs autorisees et plages numeriques plausibles."),
        ("Decision", "Sans champion, le premier modele est entraine. Avec un champion, le lot est compare a la reference : pas de changement implique NO_RETRAIN, une derive implique un entrainement."),
        ("Entrainement", "Les donnees sont nettoyees, separees en train/validation/test, puis un Random Forest est entraine."),
        ("Comparaison", "Le candidat et le champion sont evalues sur le meme jeu de test. Seul un resultat strictement meilleur est promu."),
        ("Service", "Le champion est charge par l'API. Chaque prediction est mesuree pour le monitoring."),
    ]
    for title, text in steps:
        story.append(p(f"<b>{title}.</b> {text}", style["body"]))
    story.append(PageBreak())

    story += [p("4. Demonstration realisee", style["h1"]), p("Les commandes ci-dessous ont permis de valider les comportements essentiels du projet. Elles constituent egalement un scenario de demonstration pour une presentation orale.", style["body"])]
    evidence = [
        ["Scenario", "Commande / lot", "Resultat observe", "Interpretation"],
        ["Plateforme", "docker compose up -d --build", "MLflow, API, Prometheus et Grafana healthy", "Les services sont disponibles."],
        ["Premier entrainement", "batch_initial.csv", "SUCCESS_PROMOTED", "Un modele champion est charge par l'API."],
        ["Lot normal", "batch_normal.csv", "NO_RETRAIN", "Pas de changement significatif : aucun calcul inutile."],
        ["Schema invalide", "batch_broken_extra_column.csv", "STOPPED_SCHEMA_SKEW", "La colonne promo_code est detectee et le pipeline s'arrete."],
        ["Donnees derivees", "batch_drift.csv", "SUCCESS_NOT_PROMOTED", "Une nouvelle version est creee, mais le champion reste en place car elle n'est pas meilleure."],
    ]
    story += [make_table(evidence, [2.8 * cm, 3.3 * cm, 3.25 * cm, 6.25 * cm]), Spacer(1, 0.55 * cm), p("5. Ce que montrent les interfaces", style["h1"]), p("Dans MLflow, on consulte les runs du pipeline, les metriques et les versions du modele telco-churn-classifier. L'alias champion indique le modele actuellement utilise par l'API.", style["body"]), p("Dans Prometheus, la cible fastapi est UP lorsque l'API est joignable. Dans Grafana, le dashboard MLOps - Telco Churn API affiche le nombre de predictions, leur rapidite, les erreurs de saisie, les erreurs serveur, la version chargee et la derive des donnees.", style["body"]), PageBreak()]

    story += [p("6. Commandes de demonstration", style["h1"]), p("Verifier l'etat des services :", style["h2"]), p("<font face='Courier'>docker compose ps</font>", style["body"]), p("Verifier que l'API est disponible :", style["h2"]), p("<font face='Courier'>Invoke-RestMethod http://localhost:8000/health</font>", style["body"]), p("Executer un lot normal :", style["h2"]), p("<font face='Courier'>docker compose --profile pipeline run --rm pipeline --input /data/batches/batch_normal.csv</font>", style["body"]), p("Generer du trafic pour les graphiques :", style["h2"]), p("<font face='Courier'>python scripts/send_traffic.py --url http://localhost:8000 --source data/batches/batch_normal.csv --rate 5 --duration 120 --invalid-fraction 0.05 --seed 42</font>", style["body"]), p("7. Conclusion", style["h1"]), p("Le projet fournit un cycle MLOps complet. Il protege l'entrainement contre des donnees incorrectes, detecte les changements dans les donnees, garde un historique reproductible des modeles, sert le meilleur modele par API et rend l'etat du systeme visible en temps reel.", style["body"]), p("Statut de la demonstration : services sains, modele champion charge et trois cas verifies : normal, schema invalide et derive.", style["center"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
