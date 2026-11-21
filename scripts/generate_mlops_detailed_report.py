"""Generate the detailed French MLOps Telco Churn PDF report."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Flowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scripts.generate_mlops_report import BLUE, LIGHT, MUTED, NAVY, ORANGE, RED, TEAL, footer, make_table, p, styles

OUTPUT = Path("output/pdf/rapport_mlops_telco_churn_detaille.pdf")
REGISTRY_IMAGE = Path("C:/Users/doha/AppData/Local/Temp/codex-clipboard-13aa098f-aaba-41b2-b5bf-b22041334293.png")


class DetailedPipelineDiagram(Flowable):
    """Clear end-to-end MLOps flow diagram with decision paths."""

    def __init__(self) -> None:
        super().__init__()
        self.width = 17.0 * cm
        self.height = 11.0 * cm

    def box(self, c: object, x: float, y: float, w: float, h: float, title: str, body: str, color: colors.Color) -> None:
        """Draw one labelled rounded process node."""
        c.setFillColor(colors.white)
        c.setStrokeColor(color)
        c.setLineWidth(1.25)
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 7.3)
        c.drawCentredString(x + w / 2, y + h - 12, title)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.25)
        for index, line in enumerate(body.split("\n")):
            c.drawCentredString(x + w / 2, y + h - 24 - 7.3 * index, line)

    def decision(self, c: object, x: float, y: float, size: float, label: str, color: colors.Color) -> None:
        """Draw a diamond decision node."""
        path = c.beginPath()
        path.moveTo(x + size / 2, y + size)
        path.lineTo(x + size, y + size / 2)
        path.lineTo(x + size / 2, y)
        path.lineTo(x, y + size / 2)
        path.close()
        c.setFillColor(colors.white)
        c.setStrokeColor(color)
        c.setLineWidth(1.2)
        c.drawPath(path, fill=1, stroke=1)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 6.2)
        for index, line in enumerate(label.split("\n")):
            c.drawCentredString(x + size / 2, y + size / 2 + 4 - index * 7, line)

    def arrow(self, c: object, x1: float, y1: float, x2: float, y2: float, label: str = "", color: colors.Color = NAVY) -> None:
        """Draw an arrow and an optional short decision label."""
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.15)
        c.line(x1, y1, x2, y2)
        c.circle(x2, y2, 2.15, fill=1, stroke=0)
        if label:
            c.setFont("Helvetica-Bold", 6.1)
            c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 4, label)

    def lane_title(self, c: object, x: float, text: str) -> None:
        """Draw a section heading inside the diagram."""
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 7.3)
        c.drawString(x, self.height - 15, text)

    def draw(self) -> None:
        """Render the complete data-to-monitoring process."""
        c = self.canv
        w, h = self.width, self.height
        c.setFillColor(LIGHT)
        c.roundRect(0, 0, w, h, 11, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#D8E3EC"))
        c.line(0, h - 24, w, h - 24)
        self.lane_title(c, 12, "1. CONTROLE DES DONNEES")
        self.lane_title(c, 180, "2. MODELE ET REGISTRY")
        self.lane_title(c, 374, "3. SERVICE ET SUIVI")

        self.box(c, 10, 222, 62, 39, "LOT CSV", "Donnees clients\nbrutes", BLUE)
        self.box(c, 82, 222, 82, 39, "DATA VALIDATION", "Schema + valeurs\n+ test de derive", ORANGE)
        self.decision(c, 177, 220, 46, "Schema\nvalide ?", ORANGE)
        self.arrow(c, 72, 241, 82, 241)
        self.arrow(c, 164, 241, 177, 241)
        self.box(c, 145, 139, 105, 36, "ALERTE DATA SCIENTIST", "STOPPED_SCHEMA_SKEW\nType + colonnes + details", RED)
        self.arrow(c, 200, 220, 200, 175, "non", RED)

        self.decision(c, 245, 220, 46, "Champion\nexiste ?", BLUE)
        self.arrow(c, 223, 243, 245, 243, "oui")
        self.box(c, 305, 222, 74, 39, "TEST DE DERIVE", "Reference champion\nvs lot courant", BLUE)
        self.arrow(c, 291, 243, 305, 243, "oui")
        self.box(c, 382, 139, 80, 36, "NO_RETRAIN", "Aucun\nreentrainement", TEAL)
        self.arrow(c, 342, 222, 422, 175, "non", TEAL)

        self.box(c, 252, 145, 80, 30, "REENTRAINER", "Decision TRAIN", TEAL)
        self.arrow(c, 268, 220, 292, 175, "non", TEAL)
        self.arrow(c, 342, 222, 292, 175, "oui", TEAL)
        self.box(c, 195, 75, 75, 40, "PREPARATION", "Nettoyage, split\ntrain / val / test", TEAL)
        self.box(c, 285, 75, 70, 40, "ENTRAINEMENT", "Random Forest\n+ evaluation", TEAL)
        self.box(c, 370, 75, 75, 40, "MLFLOW REGISTRY", "Nouvelle version\nmetriques + artefacts", BLUE)
        self.arrow(c, 292, 145, 232, 115)
        self.arrow(c, 270, 95, 285, 95)
        self.arrow(c, 355, 95, 370, 95)

        self.box(c, 255, 12, 95, 38, "MODEL VALIDATION", "Comparer candidat\nau champion", NAVY)
        self.decision(c, 365, 10, 42, "Nouveau\nmeilleur ?", NAVY)
        self.arrow(c, 407, 75, 302, 50)
        self.box(c, 417, 12, 57, 38, "API + SUIVI", "Champion, API\nPrometheus, Grafana", TEAL)
        self.arrow(c, 407, 31, 417, 31, "oui", TEAL)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.2)
        c.drawRightString(474, 3, "non : version conservee dans MLflow, champion inchange")


def note_box(text: str) -> Table:
    """Create a highlighted explanatory note."""
    note_style = ParagraphStyle("Note", fontName="Helvetica", fontSize=8.7, leading=12.2, textColor=colors.HexColor("#263238"))
    table = Table([[Paragraph(text, note_style)]], colWidths=[16.6 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF4F8")),
        ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def build() -> Path:
    """Create a detailed six-page professional report."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style = styles()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.65 * cm, bottomMargin=1.7 * cm, title="Rapport detaille MLOps - Telco Churn", author="Projet MLOps")
    story: list[object] = []

    story += [Spacer(1, 0.25 * cm), p("RAPPORT DETAILLE DE PROJET", style["subtitle"]), Spacer(1, 0.18 * cm), p("Pipeline MLOps - Telco Churn", style["title"]), p("Qualite des donnees, metadata management, registry, API et monitoring", style["subtitle"]), Spacer(1, 0.6 * cm), p("Vue globale du systeme", style["h1"]), p("Le flux commence par un lot de donnees client. Les controles decident ensuite s'il faut arreter, ne rien faire ou entrainer une nouvelle version de modele.", style["body"]), DetailedPipelineDiagram(), p("Figure 1 - Processus de decision et cycle de vie du modele", style["center"]), PageBreak()]

    story += [p("1. Objectif et architecture", style["h1"]), p("Le projet predit le churn : la probabilite qu'un client telecom quitte son operateur. La valeur du projet vient de l'industrialisation : le modele est controle, trace, compare, deploye et surveille.", style["body"]), p("Les quatre services Docker fonctionnent ensemble mais ont chacun une responsabilite precise.", style["body"])]
    components = [
        ["Composant", "Responsabilite", "Ce que le professeur peut verifier"],
        ["MLflow", "Historique des runs, versions de modeles, metriques et artefacts.", "Experiences, versions du registry et alias champion."],
        ["Pipeline", "Valide un lot, decide de reentrainer, entraine et compare les modeles.", "Statuts finaux : NO_RETRAIN, STOPPED_SCHEMA_SKEW, SUCCESS_PROMOTED ou SUCCESS_NOT_PROMOTED."],
        ["API FastAPI", "Expose le modele champion avec un endpoint /predict.", "GET /health et prediction de churn en JSON."],
        ["Prometheus + Grafana", "Collectent et affichent les indicateurs de fonctionnement et de derive.", "Cible fastapi UP et dashboard MLOps - Telco Churn API."],
    ]
    story += [make_table(components, [3.0 * cm, 7.2 * cm, 6.4 * cm]), Spacer(1, 0.55 * cm), note_box("<b>Idee centrale :</b> le pipeline ne remplace pas automatiquement le modele. Il le fait seulement si les donnees le justifient et si le nouveau candidat est meilleur que le champion."), Spacer(1, 0.35 * cm), p("2. Validation des donnees : premiere barriere de securite", style["h1"]), p("Chaque fichier CSV est lu comme un lot brut. Avant le nettoyage ou l'entrainement, il est controle. Cette etape evite qu'un modele apprenne a partir de donnees mal structurees ou incoherentes.", style["body"]), PageBreak()]

    story += [p("3. Data schema skew et alerte data scientist", style["h1"]), p("Un schema skew signifie que la structure ou les valeurs du lot entrant ne correspondent pas au contrat attendu. Le pipeline s'arrete immediatement, cree une alerte et explique le probleme au data scientist.", style["body"])]
    anomalies = [
        ["Type d'anomalie", "Signification", "Exemple concret", "Action du systeme"],
        ["unexpected_features", "Une colonne non prevue apparait.", "promo_code", "Arret du pipeline ; alerte avec le nom de la colonne."],
        ["missing_features", "Une colonne obligatoire est absente.", "Contract absent", "Arret du pipeline ; alerte avec la colonne manquante."],
        ["unexpected_values", "Une valeur est invalide ou hors plage autorisee.", "PaymentMethod = Bitcoin ou tenure = -5", "Arret du pipeline ; alerte avec exemples de valeurs fautives."],
    ]
    story += [make_table(anomalies, [3.6 * cm, 4.4 * cm, 4.4 * cm, 4.2 * cm]), Spacer(1, 0.42 * cm), p("Exemple observe pendant la demonstration", style["h2"]), note_box("<b>SCHEMA SKEW DETECTED : data scientist intervention required</b><br/>Input : batch_broken_extra_column.csv<br/>unexpected_features | ['promo_code'] | {'count': 1}<br/><br/>Le statut final devient <b>STOPPED_SCHEMA_SKEW</b>. Aucun nettoyage, aucun entrainement et aucune nouvelle version de modele ne sont produits."), Spacer(1, 0.45 * cm), p("4. Metadata management : la memoire de chaque execution", style["h1"]), p("Le metadata management est distinct du model registry. Le registry decrit les modeles ; les metadonnees de pipeline decrivent chaque execution, y compris celles qui ne produisent aucun modele.", style["body"])]
    metadata = [
        ["Categorie", "Exemples enregistres dans MLflow - pipeline-executions"],
        ["Contexte", "Chemin et hash du fichier, nombre de lignes, utilisateur, machine, environnement, heures de debut et de fin."],
        ["Decision", "Raison de la decision, champion precedent, nouvelle version eventuelle et indicateur de promotion."],
        ["Performance", "Duree de validation, preparation, entrainement, evaluation, registry et duree totale."],
        ["Preuves", "Rapport de validation, alerte JSON, comparaison candidat/champion et traceback en cas d'echec."],
    ]
    story += [make_table(metadata, [3.3 * cm, 13.3 * cm]), PageBreak()]

    story += [p("5. Entrainement, evaluation et Model Registry", style["h1"]), p("Lorsqu'il n'existe pas encore de champion, ou lorsqu'une derive est detectee, le pipeline nettoie le lot et le separe en train, validation et test. Un pipeline scikit-learn contient le pretraitement et un Random Forest.", style["body"])]
    lifecycle = [
        ["Etape", "Ce qui est fait", "Trace produite"],
        ["Preparation", "Suppression des doublons, conversion de TotalCharges, encodage de Churn, creation de tenure_bin et split stratifie.", "Jeux train, validation et test."],
        ["Evaluation", "Accuracy, precision, recall, F1 et ROC-AUC sur train, validation et test.", "Metriques MLflow."],
        ["Model validation", "Le candidat est compare au champion sur le meme jeu de test du lot courant.", "comparison.json et raison de promotion."],
        ["Registry", "Chaque candidat devient une nouvelle version, meme lorsqu'il n'est pas promu.", "Alias champion seulement si le candidat est meilleur."],
    ]
    story += [make_table(lifecycle, [3.0 * cm, 8.0 * cm, 5.6 * cm]), Spacer(1, 0.52 * cm), p("Resultats observes", style["h2"]), p("Le premier lot a produit un champion. Le lot normal a retourne <b>NO_RETRAIN</b>. Le lot derive a produit une nouvelle version mais le statut <b>SUCCESS_NOT_PROMOTED</b> a conserve le champion existant, car le nouveau modele n'etait pas strictement meilleur.", style["body"]), note_box("<b>Regle de promotion :</b> une version est promue uniquement si la metrique principale du candidat est strictement superieure a celle du champion, en tenant compte du seuil d'amelioration configure."), PageBreak()]

    story += [p("6. Model Registry : informations conservees", style["h1"]), p("Le Model Registry ne conserve pas seulement un fichier de modele. Il rend chaque version comprehensible et comparable : avec quel logiciel elle a ete produite, sur quelles donnees, avec quelles metriques et quel artefact de modele.", style["body"])]
    if not REGISTRY_IMAGE.exists():
        raise FileNotFoundError(f"Registry illustration not found: {REGISTRY_IMAGE}")
    image = Image(str(REGISTRY_IMAGE), width=16.2 * cm, height=7.21 * cm)
    image.hAlign = "CENTER"
    story += [image, p("Figure 2 - Les quatre categories de preuves associees a une version de modele.", style["center"]), Spacer(1, 0.25 * cm)]
    registry_contents = [
        ["Categorie", "Contenu enregistre dans ce projet"],
        ["Software", "Commit Git, versions Python / scikit-learn / MLflow, requirements, Dockerfile et pip freeze."],
        ["Data", "Lot CSV source, splits train/validation/test, hash SHA-256 et reference features pour la derive."],
        ["Metrics", "Parametres, accuracy, precision, recall, F1, ROC-AUC, comparaison au champion et delta."],
        ["Models", "Pipeline scikit-learn complet, signature, exemple d'entree et version enregistree dans MLflow."],
    ]
    story += [make_table(registry_contents, [3.0 * cm, 13.6 * cm]), Spacer(1, 0.3 * cm), p("<b>Ressource de reference.</b> La conception s'appuie sur la documentation officielle Google Cloud : <font color='#1769AA'>MLOps: Continuous delivery and automation pipelines in machine learning</font>, notamment les notions de validation des donnees et des modeles, metadata management, serving et monitoring.<br/><font size='7.5'>https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#challenges_2</font>", style["small"]), PageBreak()]

    story += [p("7. Ce que montrent Prometheus et Grafana", style["h1"])]
    monitoring = [
        ["Indicateur", "Question a laquelle il repond"],
        ["Requetes par seconde", "Combien de predictions arrivent a l'API ?"],
        ["Latence P95 / P99", "Les predictions sont-elles suffisamment rapides ?"],
        ["Erreurs 4xx", "Les utilisateurs envoient-ils des donnees incompletes ou invalides ?"],
        ["Erreurs 5xx", "L'API a-t-elle rencontre une erreur interne ?"],
        ["model_loaded et model_version_info", "Un modele est-il charge et quelle version sert les predictions ?"],
        ["Data drift", "Les demandes recentes ressemblent-elles encore aux donnees de reference ?"],
    ]
    story += [make_table(monitoring, [5.4 * cm, 11.2 * cm]), Spacer(1, 0.45 * cm), p("8. Recette de demonstration pour le professeur", style["h1"])]
    demo = [
        ["Etape", "Commande / ecran a montrer", "Preuve attendue"],
        ["1. Services", "docker compose ps", "Les quatre services sont healthy."],
        ["2. API", "Invoke-RestMethod http://localhost:8000/health", "status = ok et model_loaded = True."],
        ["3. Lot normal", "batch_normal.csv", "NO_RETRAIN : pas de calcul inutile."],
        ["4. Lot invalide", "batch_broken_extra_column.csv", "STOPPED_SCHEMA_SKEW et explication promo_code."],
        ["5. Derive", "batch_drift.csv", "Nouvelle version enregistree puis promotion ou conservation du champion."],
        ["6. Interfaces", "MLflow, Prometheus, Grafana", "Historique, cible fastapi UP et graphiques."],
    ]
    story += [make_table(demo, [2.4 * cm, 7.3 * cm, 6.9 * cm]), Spacer(1, 0.45 * cm), p("Conclusion", style["h1"]), p("Le projet repond au cycle MLOps complet : il controle les donnees, documente chaque execution, conserve toutes les versions, deploye seulement le meilleur modele et rend le systeme observable en production.", style["body"]), p("Statut actuel : plateforme Docker saine, API avec champion charge, cas normal, schema invalide et derive verifies.", style["center"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
