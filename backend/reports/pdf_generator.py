"""
Scientific PDF Report Generator
Generates comprehensive research reports for Quantum KNN vs Classical KNN benchmarks.
"""
from pathlib import Path
from typing import Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from backend.services.dataset_service import DatasetService

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    @staticmethod
    def generate_pdf_report(dataset_id: str) -> Path:
        """
        Creates a PDF research report covering dataset profiling, preprocessing,
        actual PennyLane QKNN metrics, Classical KNN comparison, and limitations.
        """
        session = DatasetService.get_session(dataset_id)
        if not session.get("trained") or not session.get("trained_models"):
            raise ValueError("Model must be trained before generating an evaluation report.")

        overview = DatasetService.get_overview(dataset_id)
        profile = DatasetService.generate_profile(dataset_id)
        trained_res = session["trained_models"]["results"]
        c_metrics = trained_res["classical_knn"]
        q_metrics = trained_res["quantum_knn"]
        circuit_info = trained_res["circuit_summary"]

        pdf_path = REPORTS_DIR / f"{dataset_id}_quantum_knn_report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            alignment=0,
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#0891b2"),
            spaceAfter=15,
        )
        h1_style = ParagraphStyle(
            "DocH1",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )
        mono_style = ParagraphStyle(
            "DocMono",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#0369a1"),
        )

        story = []

        # 1. Header
        story.append(Paragraph("Quantum Machine Learning Research Report", title_style))
        story.append(Paragraph(
            "Quantum k-Nearest Neighbour Classifier: Hybrid Feature Map Investigation (PennyLane default.qubit)",
            subtitle_style
        ))
        story.append(Spacer(1, 10))

        # 2. Dataset Overview Table
        story.append(Paragraph("1. Dataset Profile & Quality Analysis", h1_style))
        overview_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Dataset Name", overview["filename"], "Total Rows", str(overview["total_rows"])],
            ["Total Features", str(overview["total_columns"]), "Numerical Columns", str(len(overview["numerical_columns"]))],
            ["Missing Values", str(overview["missing_value_count"]), "Duplicate Records", str(overview["duplicate_record_count"])],
            ["Target Column", trained_res["target_column"], "Classes", ", ".join(trained_res["classes"])],
            ["Qubits Used", str(trained_res["qubits_used"]), "Sampling Applied", "Yes" if session["preprocessed_data"]["sample_reduced"] else "No"],
        ]
        t_overview = Table(overview_data, colWidths=[120, 140, 120, 140])
        t_overview.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_overview)
        story.append(Spacer(1, 12))

        # 3. Model Benchmark Results Table
        story.append(Paragraph("2. Performance Benchmark: Classical KNN vs. Quantum KNN", h1_style))
        story.append(Paragraph(
            "Both models evaluated on an identical stratified 80:20 train/test split. Preprocessing fitted strictly on training data without leakage.",
            body_style
        ))

        metrics_data = [
            ["Evaluation Metric", "Classical KNN (Baseline)", "Quantum KNN (PennyLane)", "Empirical Delta"],
            ["Test Accuracy", f"{c_metrics['accuracy']*100:.2f}%", f"{q_metrics['accuracy']*100:.2f}%", f"{(q_metrics['accuracy']-c_metrics['accuracy'])*100:+.2f}%"],
            ["Macro Precision", f"{c_metrics['precision_macro']*100:.2f}%", f"{q_metrics['precision_macro']*100:.2f}%", f"{(q_metrics['precision_macro']-c_metrics['precision_macro'])*100:+.2f}%"],
            ["Macro Recall", f"{c_metrics['recall_macro']*100:.2f}%", f"{q_metrics['recall_macro']*100:.2f}%", f"{(q_metrics['recall_macro']-c_metrics['recall_macro'])*100:+.2f}%"],
            ["Macro F1-Score", f"{c_metrics['f1_macro']*100:.2f}%", f"{q_metrics['f1_macro']*100:.2f}%", f"{(q_metrics['f1_macro']-c_metrics['f1_macro'])*100:+.2f}%"],
            ["Weighted F1-Score", f"{c_metrics['f1_weighted']*100:.2f}%", f"{q_metrics['f1_weighted']*100:.2f}%", f"{(q_metrics['f1_weighted']-c_metrics['f1_weighted'])*100:+.2f}%"],
            ["Training Latency", f"{c_metrics['train_time_sec']:.3f} s", f"{q_metrics['train_time_sec']:.3f} s", f"{q_metrics['train_time_sec']-c_metrics['train_time_sec']:+.3f} s"],
            ["Inference Latency", f"{c_metrics['inference_time_sec']:.3f} s", f"{q_metrics['inference_time_sec']:.3f} s", f"{q_metrics['inference_time_sec']-c_metrics['inference_time_sec']:+.3f} s"],
        ]
        t_metrics = Table(metrics_data, colWidths=[140, 130, 130, 120])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 12))

        # 4. Quantum Circuit Details
        story.append(Paragraph("3. Quantum Circuit & State Encoding Architecture", h1_style))
        circuit_table = [
            ["Feature Map Architecture", circuit_info["feature_map"].upper(), "Qubit Count", str(circuit_info["n_qubits"])],
            ["Total Gate Operations", str(circuit_info["total_gates"]), "CNOT Entangling Gates", str(circuit_info["cnot_gates"])],
            ["Entanglement Active", "Yes (Entangled State)" if circuit_info["has_entanglement"] else "No (Product State)", "Distance Metric", "Fidelity Overlap: 1 - |<ψ|φ>|²"],
        ]
        t_circ = Table(circuit_table, colWidths=[140, 120, 140, 120])
        t_circ.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_circ)
        story.append(Spacer(1, 8))

        # Circuit ASCII
        if circuit_info.get("ascii_diagram"):
            story.append(Paragraph("Quantum Feature Map Schematic:", body_style))
            # Limit ASCII length for PDF neatness
            ascii_lines = circuit_info["ascii_diagram"].split("\n")[:8]
            clean_ascii = "<br/>".join([line.replace(" ", "&nbsp;") for line in ascii_lines])
            story.append(Paragraph(clean_ascii, mono_style))
            story.append(Spacer(1, 10))

        # 5. Scientific Findings & Hardware Disclosure
        story.append(Paragraph("4. Scientific Findings & Hardware Disclosure", h1_style))
        story.append(Paragraph(trained_res["hardware_disclosure"], body_style))
        story.append(Paragraph(
            "<b>Key Observation:</b> Quantum KNN calculates neighbor proximity through inner product overlap in $2^N$-dimensional Hilbert space. "
            "On complex topological distributions, quantum entangling maps can capture non-linear feature correlations, while on linearly separable data, "
            "classical and quantum classifiers exhibit comparable decision boundaries.",
            body_style
        ))

        # Build PDF
        doc.build(story)
        session["report_path"] = str(pdf_path)
        return pdf_path
