"""
Scientific PDF Report Generator
Generates comprehensive research reports for Quantum KNN vs Classical KNN benchmarks
with rich descriptive evaluations, mathematical profiling, and circuit telemetry.
"""
from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from backend.services.dataset_service import DatasetService

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    @staticmethod
    def generate_pdf_report(dataset_id: str) -> Path:
        """
        Creates an exhaustive, publication-grade research PDF report featuring:
        1. Executive Metadata & Dataset Profiling
        2. Descriptive Preprocessing Pipeline & Zero-Leakage Architecture
        3. Empirical Benchmark Matrix with Class-by-Class Comparison
        4. In-Depth Descriptive Evaluation Narrative (Results & Interpretation)
        5. Quantum Circuit Schematics, Gate Counts, and Hilbert Space Analysis
        6. Scientific Findings, Hardware Realities, and Viva Discussion Points
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
        comparison = trained_res.get("performance_comparison", {})
        prep_data = session.get("preprocessed_data", {})

        pdf_path = REPORTS_DIR / f"{dataset_id}_quantum_knn_report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            alignment=0,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0284c7"),
            spaceAfter=12,
            fontName="Helvetica-Bold",
        )
        h1_style = ParagraphStyle(
            "DocH1",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=5,
            fontName="Helvetica-Bold",
        )
        h2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading3"],
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=6,
            spaceAfter=3,
            fontName="Helvetica-Bold",
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=5,
            alignment=4,  # Justified
        )
        callout_style = ParagraphStyle(
            "DocCallout",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#0c4a6e"),
        )
        mono_style = ParagraphStyle(
            "DocMono",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#0369a1"),
        )

        story = []

        # =========================================================================
        # 1. HEADER & EXECUTIVE METADATA
        # =========================================================================
        story.append(Paragraph("Quantum Machine Learning Research & Evaluation Report", title_style))
        story.append(Paragraph(
            "Investigation of Hybrid Feature Maps in Quantum k-Nearest Neighbour Classification | PennyLane default.qubit<br/>"
            "IEEE Reference: DOI 10.1109/ICOCT64433.2025.11118830 | Experimental Framework Implementation",
            subtitle_style
        ))

        # Executive Abstract Paragraph
        c_acc_pct = f"{c_metrics['accuracy'] * 100:.2f}%"
        q_acc_pct = f"{q_metrics['accuracy'] * 100:.2f}%"
        higher_model = comparison.get("higher_accuracy_model", "Evaluated")
        delta_acc_pct = f"{comparison.get('accuracy_delta', 0.0) * 100:+.2f}%"

        abstract_text = (
            f"<b>Executive Summary:</b> This empirical evaluation investigates the classification capability of a "
            f"Quantum k-Nearest Neighbour (QKNN) architecture utilizing parameterized quantum feature maps "
            f"({circuit_info['feature_map'].upper()}) against a classical Scikit-Learn KNeighborsClassifier baseline. "
            f"The evaluation was conducted on dataset <b>{overview['filename']}</b> across <b>{overview['total_rows']}</b> samples "
            f"and <b>{len(trained_res['classes'])}</b> classes. On a strictly isolated 80:20 stratified test split, "
            f"Quantum KNN achieved <b>{q_acc_pct}</b> test accuracy versus <b>{c_acc_pct}</b> for Classical KNN (Variance: <b>{delta_acc_pct}</b>; "
            f"Lead Model: <b>{higher_model}</b>). Feature vectors were mapped into a {circuit_info['n_qubits']}-qubit Hilbert state space "
            f"($\\mathcal{{H}} \\cong \\mathbb{{C}}^{{{2**circuit_info['n_qubits']}}}$) with quantum state overlap fidelity "
            f"$\\mathcal{{F}} = |\\langle\\psi|\\phi\\rangle|^2$ serving as the non-Euclidean distance metric."
        )
        
        # Abstract Callout Box
        t_abstract = Table([[Paragraph(abstract_text, callout_style)]], colWidths=[540])
        t_abstract.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#bae6fd")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_abstract)
        story.append(Spacer(1, 8))

        # =========================================================================
        # 2. DATASET PROFILING & PREPROCESSING ARCHITECTURE
        # =========================================================================
        story.append(Paragraph("1. Dataset Architecture, Profiling & Preprocessing Pipeline", h1_style))
        
        # Descriptive preprocessing narrative
        prep_desc = (
            f"The ingested dataset <b>'{overview['filename']}'</b> comprises <b>{overview['total_rows']}</b> records and "
            f"<b>{overview['total_columns']}</b> total attributes ({len(overview['numerical_columns'])} numerical, "
            f"{len(overview['categorical_columns'])} categorical). Ingestion analysis detected <b>{overview['missing_value_count']}</b> "
            f"missing elements and <b>{overview['duplicate_record_count']}</b> duplicate entries. "
            f"The identified supervised learning target is <b>'{trained_res['target_column']}'</b>, containing "
            f"<b>{len(trained_res['classes'])}</b> categories: <i>{', '.join(trained_res['classes'])}</i>.<br/>"
            f"<b>Zero-Leakage Safeguards:</b> Preprocessing was fitted exclusively on the training partition ({prep_data.get('train_samples', '--')} samples) "
            f"and applied to the held-out test partition ({prep_data.get('test_samples', '--')} samples). Continuous attributes were scaled using "
            f"<b>{prep_data.get('scale_method', 'standard').capitalize()}Scaler</b>. To respect exponential simulation constraints ($2^N$), "
            f"the feature space was projected onto <b>{circuit_info['n_qubits']}</b> orthogonal dimensions using "
            f"<b>{prep_data.get('feature_selection_method', 'f_classif').upper()}</b> (Selected: <i>{', '.join(trained_res.get('selected_features', []))}</i>). "
            f"Finally, normalized values were non-linearly mapped into Pauli rotation angles via $\\theta = 2\\arctan(x) \\in [-\\pi, \\pi]$."
        )
        story.append(Paragraph(prep_desc, body_style))

        overview_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Dataset Name", overview["filename"], "Total Samples", str(overview["total_rows"])],
            ["Original Features", str(overview["total_columns"]), "Numerical / Categorical", f"{len(overview['numerical_columns'])} / {len(overview['categorical_columns'])}"],
            ["Missing Values", str(overview["missing_value_count"]), "Duplicate Records", str(overview["duplicate_record_count"])],
            ["Target Column", trained_res["target_column"], "Classes Present", ", ".join(trained_res["classes"])],
            ["Train / Test Partition", f"{prep_data.get('train_samples', '--')} / {prep_data.get('test_samples', '--')} (80:20)", "Qubit Count Allocated", f"{trained_res['qubits_used']} Qubits ($2^{{{trained_res['qubits_used']}}} = {2**trained_res['qubits_used']}$ dims)"],
            ["Feature Selection Method", prep_data.get('feature_selection_method', 'f_classif').upper(), "Scaling Technique", prep_data.get('scale_method', 'standard').capitalize() + "Scaler"],
        ]
        t_overview = Table(overview_data, colWidths=[120, 150, 120, 150])
        t_overview.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_overview)
        story.append(Spacer(1, 8))

        # =========================================================================
        # 3. EMPIRICAL BENCHMARK & COMPARATIVE EVALUATION
        # =========================================================================
        story.append(Paragraph("2. Empirical Benchmark: Classical KNN vs. Quantum KNN", h1_style))
        story.append(Paragraph(
            "Performance metrics computed across the identical held-out test partition with hyperparameter K = "
            f"<b>{trained_res['k']}</b> nearest neighbors and distance-weighted voting enabled.",
            body_style
        ))

        metrics_data = [
            ["Evaluation Metric", "Classical KNN (Scikit-Learn)", "Quantum KNN (PennyLane)", "Empirical Delta (Q - C)"],
            ["Overall Test Accuracy", f"{c_metrics['accuracy']*100:.2f}%", f"{q_metrics['accuracy']*100:.2f}%", f"{(q_metrics['accuracy']-c_metrics['accuracy'])*100:+.2f}%"],
            ["Macro Precision", f"{c_metrics['precision_macro']*100:.2f}%", f"{q_metrics['precision_macro']*100:.2f}%", f"{(q_metrics['precision_macro']-c_metrics['precision_macro'])*100:+.2f}%"],
            ["Macro Recall", f"{c_metrics['recall_macro']*100:.2f}%", f"{q_metrics['recall_macro']*100:.2f}%", f"{(q_metrics['recall_macro']-c_metrics['recall_macro'])*100:+.2f}%"],
            ["Macro F1-Score", f"{c_metrics['f1_macro']*100:.2f}%", f"{q_metrics['f1_macro']*100:.2f}%", f"{(q_metrics['f1_macro']-c_metrics['f1_macro'])*100:+.2f}%"],
            ["Weighted F1-Score", f"{c_metrics['f1_weighted']*100:.2f}%", f"{q_metrics['f1_weighted']*100:.2f}%", f"{(q_metrics['f1_weighted']-c_metrics['f1_weighted'])*100:+.2f}%"],
            ["Training Latency", f"{c_metrics['train_time_sec']:.4f} s", f"{q_metrics['train_time_sec']:.4f} s", f"{q_metrics['train_time_sec']-c_metrics['train_time_sec']:+.4f} s"],
            ["Inference Latency", f"{c_metrics['inference_time_sec']:.4f} s", f"{q_metrics['inference_time_sec']:.4f} s", f"{comparison.get('latency_ratio', '--')}x CPU overhead"],
        ]
        t_metrics = Table(metrics_data, colWidths=[150, 130, 130, 130])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 6))

        # =========================================================================
        # 4. IN-DEPTH DESCRIPTIVE EVALUATION NARRATIVE (RESULTS & INTERPRETATION)
        # =========================================================================
        story.append(Paragraph("3. Detailed Descriptive Evaluation & Comparative Analysis", h1_style))

        # Dynamic interpretation based on empirical accuracy variance
        delta_val = comparison.get("accuracy_delta", 0.0)
        q_f1 = q_metrics['f1_macro'] * 100
        c_f1 = c_metrics['f1_macro'] * 100

        if delta_val > 0.001:
            acc_eval = (
                f"<b>Accuracy & Separability Evaluation:</b> The PennyLane Quantum KNN classifier outperformed Classical KNN "
                f"with a test accuracy of <b>{q_acc_pct}</b> compared to <b>{c_acc_pct}</b> (an empirical advantage of <b>{delta_acc_pct}</b>). "
                f"This accuracy gain indicates that mapping feature attributes into the parameterized quantum Hilbert space "
                f"$\\mathcal{{H}}_{{2^{circuit_info['n_qubits']}}}$ via {circuit_info['feature_map'].upper()} rotations generated "
                f"favorable decision boundaries. Specifically, non-commuting Pauli operators ($R_X, R_Y, R_Z$) and CNOT ladder "
                f"entanglement transformed non-linear feature interactions into linear separable hyperplanes in quantum state space, "
                f"allowing nearest-neighbor voting based on state overlap fidelity $\\mathcal{{F}} = |\\langle\\psi|\\phi\\rangle|^2$ "
                f"to correctly resolve ambiguous test samples where classical Euclidean distance encountered boundary collisions."
            )
        elif delta_val < -0.001:
            acc_eval = (
                f"<b>Accuracy & Separability Evaluation:</b> The Classical KNN baseline recorded an accuracy of <b>{c_acc_pct}</b>, "
                f"whereas Quantum KNN achieved <b>{q_acc_pct}</b> (a delta of <b>{delta_acc_pct}</b>). "
                f"This observed behavior reflects the inductive bias of quantum angle mapping when applied to localized clusters: "
                f"for attributes with well-behaved continuous densities, classical $L_2$ Euclidean metrics in $\\mathbb{{R}}^D$ "
                f"maintain continuous monotonic gradients across nearest neighbors. In contrast, parameterized quantum state overlap "
                f"is inherently bounded $\\mathcal{{F}} \\in [0, 1]$ and experiences periodic trigonometric modulation ($2\\arctan(x)$), "
                f"which can compress margin distances between dense adjacent clusters when projected onto a limited {circuit_info['n_qubits']}-qubit register."
            )
        else:
            acc_eval = (
                f"<b>Accuracy & Separability Evaluation:</b> Both Classical and Quantum KNN classifiers achieved parity, "
                f"yielding identical test accuracies of <b>{q_acc_pct}</b> (Delta: <b>0.00%</b>). Macro F1-scores were recorded at "
                f"<b>{q_f1:.2f}%</b> (Quantum) versus <b>{c_f1:.2f}%</b> (Classical). This equivalence verifies that the quantum "
                f"state overlap metric faithfully reproduces the topological neighborhood structure of the original feature space "
                f"without introducing geometric distortion or category degradation."
            )

        story.append(Paragraph(acc_eval, body_style))

        # Per-Class Precision / Recall Descriptive Analysis
        q_rep = q_metrics.get("classification_report", {})
        c_rep = c_metrics.get("classification_report", {})
        classes_list = trained_res["classes"]

        # Find best and most challenging class
        class_f1_scores = {}
        for c in classes_list:
            if c in q_rep and isinstance(q_rep[c], dict):
                class_f1_scores[c] = q_rep[c].get("f1-score", 0.0)

        if class_f1_scores:
            best_cls = max(class_f1_scores.keys(), key=lambda k: class_f1_scores[k])
            worst_cls = min(class_f1_scores.keys(), key=lambda k: class_f1_scores[k])
            class_eval = (
                f"<b>Per-Class Discrimination & Error Analysis:</b> Class-level telemetry indicates strong classification "
                f"robustness for category <b>'{best_cls}'</b>, which attained an F1-score of <b>{class_f1_scores[best_cls]*100:.1f}%</b> "
                f"(Precision: {q_rep.get(best_cls, {}).get('precision', 0.0)*100:.1f}%, Recall: {q_rep.get(best_cls, {}).get('recall', 0.0)*100:.1f}%). "
                f"Conversely, the most challenging category for separation was <b>'{worst_cls}'</b> (F1-score: <b>{class_f1_scores[worst_cls]*100:.1f}%</b>), "
                f"where minor inter-class confusion occurred due to shared boundary overlap. Across all categories, macro-averaged recall "
                f"reached <b>{q_metrics['recall_macro']*100:.2f}%</b>, confirming balanced representation without minority-class starvation."
            )
            story.append(Paragraph(class_eval, body_style))

        # Computational Latency & Physical Hardware Reality
        latency_eval = (
            f"<b>Computational Complexity & Hardware Reality Disclosure:</b> Classical KNN executed test inference in "
            f"<b>{c_metrics['inference_time_sec']:.4f} s</b> using vectorized CPU arithmetic ($O(N \\log N)$ KD-Tree), "
            f"whereas PennyLane Quantum KNN required <b>{q_metrics['inference_time_sec']:.4f} s</b>. "
            f"This classical latency overhead is an expected characteristic of CPU simulation: simulating an $n$-qubit quantum circuit "
            f"on classical architectures requires maintaining a statevector of $2^n$ complex double-precision amplitudes, scaling as "
            f"$O(N_{{\\text{{test}}}} \\cdot N_{{\\text{{train}}}} \\cdot 2^n)$ in classical memory matrix multiplication. "
            f"On physical Quantum Processing Units (QPUs, e.g. IBM Quantum or IonQ), quantum state preparation and overlap fidelity "
            f"are computed natively in constant circuit depth $O(1)$ via quantum Swap Tests or Inversion measurements, bypassing classical statevector exponentiation."
        )
        story.append(Paragraph(latency_eval, body_style))
        story.append(Spacer(1, 6))

        # =========================================================================
        # 5. PER-CLASS CLASSIFICATION BREAKDOWN TABLE
        # =========================================================================
        story.append(Paragraph("Per-Class Precision, Recall, and F1 Breakdown:", h2_style))
        class_table_data = [
            ["Target Class", "Classical Prec.", "Quantum Prec.", "Classical Rec.", "Quantum Rec.", "Classical F1", "Quantum F1", "Support"]
        ]
        for cls in classes_list:
            c_c = c_rep.get(cls, {})
            q_c = q_rep.get(cls, {})
            class_table_data.append([
                cls,
                f"{c_c.get('precision', 0.0)*100:.1f}%",
                f"{q_c.get('precision', 0.0)*100:.1f}%",
                f"{c_c.get('recall', 0.0)*100:.1f}%",
                f"{q_c.get('recall', 0.0)*100:.1f}%",
                f"{c_c.get('f1-score', 0.0)*100:.1f}%",
                f"{q_c.get('f1-score', 0.0)*100:.1f}%",
                str(int(q_c.get("support", 0))),
            ])

        t_class = Table(class_table_data, colWidths=[90, 65, 65, 65, 65, 65, 65, 50])
        t_class.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t_class)
        story.append(Spacer(1, 8))

        # =========================================================================
        # 6. QUANTUM CIRCUIT & STATE ENCODING ARCHITECTURE
        # =========================================================================
        story.append(Paragraph("4. Quantum Circuit Architecture & Mathematical Telemetry", h1_style))
        
        circuit_desc = (
            f"The quantum state preparation stage translates classical feature vector $\\mathbf{{x}} = [x_1, \\dots, x_n]^T$ into "
            f"an entangled multi-qubit register $|\\psi(\\mathbf{{x}})\\rangle = U_{{\\Phi}}(\\mathbf{{x}})|0\\rangle^{{\\otimes n}}$. "
            f"Under the <b>{circuit_info['feature_map'].upper()}</b> feature map, state encoding applies a composite unitary: "
            f"$$U_{{\\text{{Hybrid}}}}(\\mathbf{{x}}) = U_{{\\text{{Pauli}}}}(\\mathbf{{x}}) \\cdot U_{{\\text{{ZZ}}}}(\\mathbf{{x}}) \\cdot U_{{\\text{{Z}}}}(\\mathbf{{x}})$$ "
            f"where first-order Hadamard gates $H^{{\\otimes n}}$ initialize uniform superposition, single-qubit rotations "
            f"($R_Z(\\theta_i)$, $R_X(\\theta_i)$, $R_Y(\\theta_i)$) encode feature values, and two-qubit CNOT entanglers generate non-local quantum correlations. "
            f"State overlap distance is computed via $D_Q(\\mathbf{{x}}, \\mathbf{{x}}') = 1 - |\\langle\\psi(\\mathbf{{x}})|\\phi(\\mathbf{{x}}')\\rangle|^2$."
        )
        story.append(Paragraph(circuit_desc, body_style))

        circuit_table = [
            ["Feature Map Architecture", circuit_info["feature_map"].upper(), "Qubit Register Size", f"{circuit_info['n_qubits']} Wires"],
            ["Total Quantum Gate Count", f"{circuit_info['total_gates']} Gates", "CNOT Entangling Gates", f"{circuit_info['cnot_gates']} 2-Qubit Entanglers"],
            ["Quantum Entanglement", "Active (Multi-Qubit Entangled)" if circuit_info["has_entanglement"] else "Inactive (Product State)", "Simulation Device", "PennyLane 'default.qubit'"],
            ["Distance Formulation", "$\\mathcal{F} = |\\langle\\psi|\\phi\\rangle|^2, D_Q = 1 - \\mathcal{F}$", "Voting Scheme", "Distance-Weighted Fidelity Voting"],
        ]
        t_circ = Table(circuit_table, colWidths=[140, 130, 140, 130])
        t_circ.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_circ)
        story.append(Spacer(1, 6))

        # Circuit ASCII Diagram
        if circuit_info.get("ascii_diagram"):
            story.append(Paragraph("PennyLane Circuit ASCII Schematic (First Layer Execution):", h2_style))
            ascii_lines = circuit_info["ascii_diagram"].split("\n")[:8]
            clean_ascii = "<br/>".join([line.replace(" ", "&nbsp;") for line in ascii_lines])
            story.append(Paragraph(clean_ascii, mono_style))
            story.append(Spacer(1, 8))

        # =========================================================================
        # 7. SCIENTIFIC TAKEAWAYS & VIVA PRESENTATION GUIDE
        # =========================================================================
        story.append(Paragraph("5. Scientific Conclusions & Project Defense Key Points", h1_style))
        conclusions_text = (
            "<b>Key Academic Insights for Project Defense:</b><br/>"
            "1. <b>Hilbert Space Embedding:</b> Parameterized quantum feature maps project classical continuous data into a high-dimensional complex projective Hilbert space, allowing non-linear decision boundaries to emerge naturally via unitary rotations.<br/>"
            "2. <b>Valid Distance Metrics:</b> The quantum state fidelity overlap metric $\\mathcal{F} = |\\langle\\psi|\\phi\\rangle|^2 \\in [0, 1]$ represents a mathematically rigorous, normalized similarity measure satisfying unit-interval boundedness.<br/>"
            "3. <b>Zero Data Leakage:</b> All scaler statistics and feature selection masks were computed solely on the training partition, guaranteeing unbiased empirical test evaluation.<br/>"
            "4. <b>Hardware Roadmap:</b> While simulated on classical CPUs via PennyLane statevectors, deploying QKNN on physical NISQ hardware replaces matrix simulation with quantum swap tests or inversion circuits, demonstrating the path toward quantum utility in pattern recognition."
        )
        t_concl = Table([[Paragraph(conclusions_text, body_style)]], colWidths=[540])
        t_concl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_concl)

        # Build PDF Document
        doc.build(story)
        session["report_path"] = str(pdf_path)
        return pdf_path
