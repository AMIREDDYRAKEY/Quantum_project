"""
Compiles the Master Overview PDF Document:
- Application Run Commands (Backend, Frontend, Tests, Deployment)
- PDF Generation Commands (CLI, Scripts, REST API, UI)
- Complete Datasets Specifications (7 Benchmark Datasets + Custom Ingestion Pipeline)
- Quantum Machine Learning Methodology (Feature Maps, Hilbert Space, Fidelity Overlap)
"""
import sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF_ROOT = ROOT_DIR / "QUANTUM_KNN_STUDIO_MASTER_OVERVIEW.pdf"
OUTPUT_PDF_REPORTS = ROOT_DIR / "backend" / "data" / "reports" / "QUANTUM_KNN_STUDIO_MASTER_OVERVIEW.pdf"
OUTPUT_PDF_REPORTS.parent.mkdir(parents=True, exist_ok=True)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Quantum KNN Studio — System Overview, Execution Commands & Datasets Guide")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        self.drawString(36, 25, "IEEE ICOCT 2025 (DOI: 10.1109/ICOCT64433.2025.11118830) | PennyLane default.qubit")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 35, 576, 35)

        self.restoreState()


def generate_overview_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubTitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0284c7"),
        fontName="Helvetica-Bold",
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica-Bold",
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
        alignment=4,
    )
    code_block_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0369a1"),
    )
    callout_style = ParagraphStyle(
        "CalloutBox",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0c4a6e"),
    )

    story = []

    # =========================================================================
    # DOCUMENT HEADER
    # =========================================================================
    story.append(Paragraph("Quantum KNN Studio: System Overview, Commands & Datasets Guide", title_style))
    story.append(Paragraph(
        "Complete Operational Reference & Benchmark Dossier | IEEE ICOCT 2025 Paper Implementation<br/>"
        "PennyLane Quantum Engine &bull; Scikit-Learn Classical Baseline &bull; React Vite Full-Stack Platform",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8))

    # Executive Summary Box
    summary_html = (
        "<b>Executive Purpose:</b> This master document serves as the unified operational, command-line, and technical "
        "dossier for the <i>Quantum k-Nearest Neighbour Classifier: Investigating Hybrid Feature Map Approach</i> project. "
        "It consolidates: <b>(1)</b> all commands required to start and build the full-stack web application, <b>(2)</b> all methods "
        "to generate official PDF research reports, <b>(3)</b> complete technical specifications for all 7 benchmark datasets plus custom ingestion, "
        "and <b>(4)</b> the mathematical foundations of the PennyLane quantum feature maps and fidelity distance metrics."
    )
    t_summary = Table([[Paragraph(summary_html, callout_style)]], colWidths=[540])
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#0284c7")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: APPLICATION EXECUTION COMMANDS
    # =========================================================================
    story.append(Paragraph("1. Commands to Run the Application & Full-Stack Services", h1_style))
    story.append(Paragraph(
        "The project consists of a Python Flask REST backend (port 5000) and a React 18 / Vite / Tailwind frontend (port 3000). "
        "Follow the steps below to start each component.",
        body_style
    ))

    # Command Table
    run_cmds = [
        ["Operation", "Platform / Shell", "Exact Execution Command", "Expected Port / Output"],
        [
            Paragraph("<b>Activate Virtual Environment</b>", body_style),
            "Windows PowerShell",
            Paragraph("<code>.\\.venv\\Scripts\\Activate.ps1</code>", code_block_style),
            "Active .venv prompt",
        ],
        [
            Paragraph("<b>Install Dependencies</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>python -m pip install -r requirements.txt</code>", code_block_style),
            "All packages installed",
        ],
        [
            Paragraph("<b>Start Backend API Server</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>python backend\\app.py</code>", code_block_style),
            Paragraph("<b>http://localhost:5000</b><br/>(Operational)", body_style),
        ],
        [
            Paragraph("<b>Start Frontend Web UI</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>cd frontend<br/>npm run dev</code>", code_block_style),
            Paragraph("<b>http://localhost:3000</b><br/>(Local React App)", body_style),
        ],
        [
            Paragraph("<b>Build Production Frontend</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>cd frontend<br/>npm run build</code>", code_block_style),
            "dist/ bundle compiled in ~10s",
        ],
        [
            Paragraph("<b>Run Backend Test Suite</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>python -m pytest backend\\tests\\ -v</code>", code_block_style),
            "46 tests passed (100%)",
        ],
        [
            Paragraph("<b>Run Frontend Tests</b>", body_style),
            "PowerShell / Bash",
            Paragraph("<code>cd frontend<br/>npm test</code>", code_block_style),
            "6 Vitest suites passed",
        ],
        [
            Paragraph("<b>Verify Backend Health</b>", body_style),
            "cURL / Browser",
            Paragraph("<code>curl http://localhost:5000/api/health</code>", code_block_style),
            '{"status": "healthy"}',
        ],
    ]
    t_cmds = Table(run_cmds, colWidths=[120, 95, 205, 120])
    t_cmds.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_cmds)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: PDF REPORT GENERATION COMMANDS
    # =========================================================================
    story.append(Paragraph("2. Commands to Generate Scientific PDF Research Reports", h1_style))
    story.append(Paragraph(
        "Comprehensive PDF evaluation reports can be compiled using <b>three independent methods</b>: "
        "via the standalone Python script (<code>generate_reports.py</code>), via HTTP REST API endpoints, or via the interactive Web Dashboard.",
        body_style
    ))

    pdf_cmds = [
        ["Generation Method", "Target Dataset", "Command / Endpoint Execution", "Output Artifact"],
        [
            Paragraph("<b>CLI: All 7 Benchmarks</b>", body_style),
            "All 7 Pre-Packaged Datasets",
            Paragraph("<code>python generate_reports.py</code>", code_block_style),
            Paragraph("7 PDFs saved to<br/><code>backend/data/reports/</code>", body_style),
        ],
        [
            Paragraph("<b>CLI: Single Dataset</b>", body_style),
            "Iris (or wine, student, etc.)",
            Paragraph("<code>python generate_reports.py --dataset iris</code>", code_block_style),
            "sample_iris_quantum_knn_report.pdf",
        ],
        [
            Paragraph("<b>CLI: Custom Upload CSV</b>", body_style),
            "User Provided CSV File",
            Paragraph("<code>python generate_reports.py --file path/to/my_data.csv --target label_col</code>", code_block_style),
            "custom_my_data_quantum_knn_report.pdf",
        ],
        [
            Paragraph("<b>REST API: Direct Download</b>", body_style),
            "sample_iris (via GET)",
            Paragraph("<code>curl -o iris_report.pdf http://localhost:5000/api/dataset/sample_iris/report</code>", code_block_style),
            "Direct application/pdf stream",
        ],
        [
            Paragraph("<b>REST API: Production URL</b>", body_style),
            "Render Cloud Backend",
            Paragraph("<code>curl -O https://quantum-project-kxpm.onrender.com/api/dataset/sample_wine/report</code>", code_block_style),
            "sample_wine_quantum_knn_report.pdf",
        ],
        [
            Paragraph("<b>Web Dashboard: 1-Click</b>", body_style),
            "Any Uploaded / Sample Dataset",
            Paragraph("Navigate to <b>Step 7: PDF Report</b> and click <b>[Download Official PDF Report]</b> or use <b>[Direct Mirror Link]</b>", body_style),
            "Browser native PDF download",
        ],
    ]
    t_pdf = Table(pdf_cmds, colWidths=[120, 105, 215, 100])
    t_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_pdf)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: BENCHMARK DATASETS COMPREHENSIVE SPECIFICATION
    # =========================================================================
    story.append(Paragraph("3. Datasets Overview & Scientific Specifications", h1_style))
    story.append(Paragraph(
        "The platform natively packages <b>7 domain-diverse benchmark classification datasets</b> spanning botanical, "
        "medical, educational, agronomic, consumer marketing, and meteorological applications. It additionally supports arbitrary custom user uploads.",
        body_style
    ))

    ds_specs = [
        ["Dataset Name", "Domain Field", "Samples", "Features", "Target Column", "Target Classes", "Physical Problem Definition"],
        [
            Paragraph("<b>Iris</b><br/>(iris.csv)", body_style),
            "Botanical Taxonomy",
            "150",
            "4 (Sepal / Petal)",
            "target",
            "3 classes (0, 1, 2)",
            "Morphological species separation: Setosa, Versicolor, Virginica.",
        ],
        [
            Paragraph("<b>Student Performance</b><br/>(student_performance.csv)", body_style),
            "Educational Analytics",
            "300",
            "6 (Hours, Attend, Grades)",
            "academic_result",
            "3 tiers (Distinction, Merit, Pass)",
            "Predicts student exam tier for proactive tutoring intervention.",
        ],
        [
            Paragraph("<b>Crop Recommendation</b><br/>(agriculture_crop.csv)", body_style),
            "Smart Agronomy",
            "300",
            "7 (N, P, K, pH, Rain)",
            "recommended_crop",
            "3 cultivars (Rice, Maize, Cotton)",
            "Recommends optimal crop to maximize harvest yield and save water.",
        ],
        [
            Paragraph("<b>Social Engagement</b><br/>(social_consumer.csv)", body_style),
            "Digital Marketing",
            "400",
            "5 (Age, Salary, Screen Time)",
            "ad_purchase_decision",
            "2 classes (Purchased: 0 / 1)",
            "Predicts consumer ad conversion to optimize digital campaign ROI.",
        ],
        [
            Paragraph("<b>Wine Cultivars</b><br/>(wine.csv)", body_style),
            "Chemical Enology",
            "178",
            "13 (Alcohol, Phenols, Flavan)",
            "target",
            "3 cultivars (Cultivar 1, 2, 3)",
            "HPLC chemical spectrometry authenticates Italian regional wine origin.",
        ],
        [
            Paragraph("<b>Rain in Australia</b><br/>(rain.csv)", body_style),
            "Meteorology",
            "300",
            "12 (Temp, Wind, Humidity)",
            "RainTomorrow",
            "2 classes (Yes / No)",
            "Forecasts next-day precipitation from regional telemetry stations.",
        ],
        [
            Paragraph("<b>Indian Liver Patient</b><br/>(indian_liver.csv)", body_style),
            "Clinical Diagnostics",
            "583",
            "10 (Bilirubin, ALT, AST, Prot)",
            "Dataset",
            "2 classes (Patient: 1 / 2)",
            "Biochemical blood screening detects liver cirrhosis and hepatitis.",
        ],
    ]
    t_ds = Table(ds_specs, colWidths=[90, 75, 40, 65, 75, 75, 120])
    t_ds.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    # Custom Upload Feature Card
    upload_card_html = (
        "<b>Custom Dataset Ingestion Pipeline:</b> Supports user file uploads in <b>CSV, XLSX, or XLS</b> formats up to <b>50 MB</b>. "
        "The automated backend executes: <b>(1)</b> Schema and null-value profiling, <b>(2)</b> Zero-leakage StandardScaler fitted strictly on training data, "
        "<b>(3)</b> SelectKBest feature-to-qubit dimension mapping via ANOVA F-statistic (mapping $D$ continuous features onto $N$ qubits without silent data dropping), "
        "and <b>(4)</b> Pauli rotation angle mapping via $\\theta = 2\\arctan(x) \\in [-\\pi, \\pi]$ for unitary quantum gates."
    )
    t_up = Table([[Paragraph(upload_card_html, callout_style)]], colWidths=[540])
    t_up.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_up)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: QUANTUM KNN VS CLASSICAL METHODOLOGY
    # =========================================================================
    story.append(Paragraph("4. Quantum Feature Maps & Mathematical Methodology", h1_style))
    story.append(Paragraph(
        "The primary scientific contribution investigated in this project is the comparative evaluation of the <b>Hybrid Feature Map</b> "
        "against individual quantum feature maps and Classical Euclidean KNN.",
        body_style
    ))

    methodology_data = [
        ["Model Architecture", "Paradigm", "Mathematical Formulation / Quantum Circuit Ansatz", "Entanglement Type"],
        [
            Paragraph("<b>Classical KNN</b><br/>(Scikit-Learn Baseline)", body_style),
            "Classical CPU",
            Paragraph("Standard Euclidean $L_2$ metric in $\\mathbb{R}^D$:<br/>$D(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^D (x_i - y_i)^2}$", code_block_style),
            "None (Euclidean coordinate space)",
        ],
        [
            Paragraph("<b>Z Feature Map QKNN</b>", body_style),
            "PennyLane Quantum",
            Paragraph("First-order Pauli-Z rotations without entanglement:<br/>$U_Z(\\mathbf{x}) = \\bigotimes_{i=1}^n R_Z(2\\arctan(x_i)) \\cdot H$", code_block_style),
            "Unentangled (Product state)",
        ],
        [
            Paragraph("<b>ZZ Feature Map QKNN</b>", body_style),
            "PennyLane Quantum",
            Paragraph("Second-order Ising phase interactions with CNOT ladders:<br/>$U_{ZZ}(\\mathbf{x}) = \\prod_{(i,j)} \\text{CNOT}_{i,j} \\cdot R_Z((\\pi-x_i)(\\pi-x_j)) \\cdot \\text{CNOT}_{i,j}$", code_block_style),
            "Pairwise CNOT 2-Qubit Entanglement",
        ],
        [
            Paragraph("<b>Pauli Feature Map QKNN</b>", body_style),
            "PennyLane Quantum",
            Paragraph("Non-commuting multi-axis parameterized unitary rotations:<br/>$U_{\\text{Pauli}}(\\mathbf{x}) = \\bigotimes_{i} [R_Z(x_i) R_Y(x_i) R_X(x_i) H]$ with ring coupling", code_block_style),
            "Multi-axis Non-commuting Coupling",
        ],
        [
            Paragraph("<b>Hybrid Feature Map QKNN</b><br/>(Core Paper Proposal)", body_style),
            "PennyLane Quantum",
            Paragraph("Deep composite cascade combining Z, ZZ, and Pauli ansatzes:<br/>$U_{\\text{Hybrid}}(\\mathbf{x}) = U_{\\text{Pauli}}(\\mathbf{x}) \\cdot U_{ZZ}(\\mathbf{x}) \\cdot U_Z(\\mathbf{x})$", code_block_style),
            "Full Multi-Layer High-Degree Entanglement",
        ],
    ]
    t_meth = Table(methodology_data, colWidths=[110, 80, 240, 110])
    t_meth.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_meth)
    story.append(Spacer(1, 8))

    # Quantum Distance Formulation
    distance_html = (
        "<b>Quantum State Overlap Distance Metric:</b> In Quantum KNN, similarity between two records $\\mathbf{x}$ and $\\mathbf{y}$ "
        "is not computed via classical Euclidean distances. Instead, it is evaluated as the <b>quantum state fidelity overlap</b>: "
        "$$\\mathcal{F}(\\mathbf{x}, \\mathbf{y}) = |\\langle\\psi(\\mathbf{x})|\\phi(\\mathbf{y})\\rangle|^2 \\in [0, 1]$$ "
        "Fidelity is converted to a normalized metric distance: "
        "$$D_Q(\\mathbf{x}, \\mathbf{y}) = 1 - \\mathcal{F}(\\mathbf{x}, \\mathbf{y})$$ "
        "The $K$ smallest quantum distances determine nearest neighbors, and distance-weighted voting $\\omega_i = 1 / (D_Q + \\epsilon)$ "
        "predicts the class label."
    )
    t_dist = Table([[Paragraph(distance_html, callout_style)]], colWidths=[540])
    t_dist.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#16a34a")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_dist)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 5: PROJECT REPOSITORY & DEPLOYMENT ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("5. Project Repository & Cloud Deployment Architecture", h1_style))
    
    repo_data = [
        ["Component", "Platform", "URL / Deployment Location", "Status"],
        [
            Paragraph("<b>Git Remote Repository</b>", body_style),
            "GitHub",
            Paragraph("<font color='#0284c7'>https://github.com/AMIREDDYRAKEY/Quantum_project.git</font>", code_block_style),
            "Branch: main (Synced)",
        ],
        [
            Paragraph("<b>Production Frontend</b>", body_style),
            "Vercel Edge Network",
            Paragraph("<font color='#0284c7'>https://quantum-project-ten.vercel.app</font>", code_block_style),
            "Automated CI/CD Deploy",
        ],
        [
            Paragraph("<b>Production Backend</b>", body_style),
            "Render Cloud Hosting",
            Paragraph("<font color='#0284c7'>https://quantum-project-kxpm.onrender.com</font>", code_block_style),
            "Gunicorn Web Service",
        ],
        [
            Paragraph("<b>Local Development</b>", body_style),
            "Localhost (Windows / Linux)",
            Paragraph("Frontend: <b>http://localhost:3000</b> | Backend: <b>http://localhost:5000</b>", code_block_style),
            "Full Development Server",
        ],
    ]
    t_repo = Table(repo_data, colWidths=[110, 105, 225, 100])
    t_repo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_repo)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Master Overview PDF built: {output_path.resolve()} ({output_path.stat().st_size / 1024:.1f} KB)")


def main():
    generate_overview_pdf(OUTPUT_PDF_ROOT)
    generate_overview_pdf(OUTPUT_PDF_REPORTS)


if __name__ == "__main__":
    main()
