#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import html
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensorqa.reporting.report_builder import ReportBundle


@dataclass(frozen=True)
class HTMLReportOptions:
    """Rendering options for a self-contained SensorQA HTML report."""

    include_metric_descriptions: bool = True
    include_analysis_metadata: bool = False
    embed_plot_images: bool = True
    maximum_embedded_image_bytes: int = 5_000_000

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(self.include_metric_descriptions, bool):
            raise TypeError("include_metric_descriptions must be a boolean.")
        if not isinstance(self.include_analysis_metadata, bool):
            raise TypeError("include_analysis_metadata must be a boolean.")
        if not isinstance(self.embed_plot_images, bool):
            raise TypeError("embed_plot_images must be a boolean.")
        if not isinstance(self.maximum_embedded_image_bytes, int):
            raise TypeError("maximum_embedded_image_bytes must be an integer.")
        if self.maximum_embedded_image_bytes < 0:
            raise ValueError("maximum_embedded_image_bytes must be nonnegative.")


class HTMLReportRenderer:
    """Render a standalone, printable SensorQA HTML report."""

    def render(
        self,
        report: ReportBundle,
        *,
        options: HTMLReportOptions | None = None,
    ) -> str:
        """Render render.
        
        Args:
            report: Report used by this function.
            options: Options used by this function.
        
        Returns:
            Requested text value.
        """
        if not isinstance(report, ReportBundle):
            raise TypeError("report must be a ReportBundle.")
        resolved = options or HTMLReportOptions()
        if not isinstance(resolved, HTMLReportOptions):
            raise TypeError("options must be HTMLReportOptions or None.")

        workflow = report.workflow
        dataset = workflow.dataset
        qualification = workflow.qualification

        body: list[str] = []
        body.append(self._header(report))
        body.append(self._executive_summary(report))

        if dataset is not None:
            body.append(self._dataset_section(report))

        if qualification is not None:
            body.append(self._qualification_section(report))

        body.append(self._analysis_section(report, resolved))

        if report.calibration_summary is not None:
            body.append(self._calibration_section(report.calibration_summary))

        body.append(self._messages_section(report))
        body.append(self._footer(report))

        return self._document("\n".join(body), title=report.metadata.title)

    def write(
        self,
        report: ReportBundle,
        destination: str | Path,
        *,
        options: HTMLReportOptions | None = None,
    ) -> Path:
        """Write write.
        
        Args:
            report: Report used by this function.
            destination: Destination used by this function.
            options: Options used by this function.
        
        Returns:
            Resolved path.
        """
        if not isinstance(destination, (str, Path)):
            raise TypeError("destination must be a string path or pathlib.Path.")
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(report, options=options), encoding="utf-8")
        return path

    def _header(self, report: ReportBundle) -> str:
        """Return header.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        meta = report.metadata
        details: list[str] = []
        if meta.subtitle:
            details.append(f'<div class="subtitle">{self._e(meta.subtitle)}</div>')
        if meta.prepared_for:
            details.append(
                f'<div><strong>Prepared for:</strong> {self._e(meta.prepared_for)}</div>'
            )
        if meta.prepared_by:
            details.append(
                f'<div><strong>Prepared by:</strong> {self._e(meta.prepared_by)}</div>'
            )
        details.append(
            f'<div><strong>Generated:</strong> {self._e(report.generated_at_utc)}</div>'
        )
        if report.source_file:
            details.append(
                f'<div><strong>Source:</strong> {self._e(report.source_file)}</div>'
            )
        return (
            '<header class="report-header">'
            f'<div class="brand">SensorQA</div><h1>{self._e(meta.title)}</h1>'
            + "".join(details)
            + "</header>"
        )

    def _executive_summary(self, report: ReportBundle) -> str:
        """Return executive summary.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        workflow = report.workflow
        qualification = workflow.qualification
        overall = qualification.overall_status if qualification else "not evaluated"
        cards = [
            self._card("Workflow", "Complete" if workflow.success else "Incomplete", workflow.success),
            self._card("Analyses", str(workflow.analysis_count), None),
            self._card("Successful", str(workflow.successful_analysis_count), True),
            self._card("Qualification", overall.replace("_", " ").title(), self._status_bool(overall)),
        ]
        if qualification is not None:
            cards.extend(
                [
                    self._card("Requirements Passed", str(qualification.passed), True),
                    self._card("Requirements Failed", str(qualification.failed), qualification.failed == 0),
                ]
            )
        return (
            '<section id="summary"><h2>Executive Summary</h2>'
            f'<div class="card-grid">{"".join(cards)}</div></section>'
        )

    def _dataset_section(self, report: ReportBundle) -> str:
        """Return dataset section.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        dataset = report.workflow.dataset
        assert dataset is not None
        sampling = dataset.sampling
        rows = [
            ("Sensor type", dataset.sensor_type or "—"),
            ("Rows", self._number(dataset.row_count)),
            ("Columns", self._number(dataset.column_count)),
            ("Validation", "Valid" if dataset.valid else "Issues detected"),
            ("Errors", str(dataset.error_count)),
            ("Warnings", str(dataset.warning_count)),
        ]
        if sampling is not None and sampling.available:
            rows.extend(
                [
                    ("Duration", self._metric(sampling.duration_seconds, "s")),
                    ("Estimated sampling rate", self._metric(sampling.estimated_sampling_rate_hz, "Hz")),
                    ("Duplicate timestamps", str(sampling.duplicate_timestamp_count)),
                    ("Large gaps", str(sampling.large_gap_count)),
                ]
            )

        units = "".join(
            f"<tr><td>{self._e(key)}</td><td>{self._e(value)}</td></tr>"
            for key, value in sorted(dataset.units.items())
        ) or '<tr><td colspan="2">No unit assignments.</td></tr>'

        issues = "".join(
            '<li>'
            f'<span class="badge badge-{self._status_class(issue.severity)}">{self._e(issue.severity)}</span> '
            f'{self._e(issue.message)}'
            "</li>"
            for issue in dataset.issues
        ) or "<li>No dataset issues reported.</li>"

        return (
            '<section id="dataset"><h2>Dataset</h2>'
            + self._key_value_table(rows)
            + '<h3>Units</h3><table><thead><tr><th>Field</th><th>Unit</th></tr></thead>'
            f'<tbody>{units}</tbody></table>'
            + f'<h3>Data Quality</h3><ul class="message-list">{issues}</ul></section>'
        )

    def _qualification_section(self, report: ReportBundle) -> str:
        """Return qualification section.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        qualification = report.workflow.qualification
        assert qualification is not None
        rows = []
        for check in qualification.checks:
            measured = self._metric(check.measured_value, check.unit)
            limit = self._metric(check.limit_value, check.unit)
            rows.append(
                "<tr>"
                f"<td>{self._e(check.tool_name)}</td>"
                f"<td>{self._e(check.metric_name)}</td>"
                f"<td>{measured}</td>"
                f"<td>{self._e(check.operator or '—')}</td>"
                f"<td>{limit}</td>"
                f'<td><span class="badge badge-{self._status_class(check.status)}">{self._e(check.status)}</span></td>'
                "</tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No requirement checks were evaluated.</td></tr>')
        return (
            '<section id="qualification"><h2>Qualification</h2>'
            f'<p>Overall status: <span class="badge badge-{self._status_class(qualification.overall_status)}">'
            f'{self._e(qualification.overall_status)}</span></p>'
            '<table><thead><tr><th>Tool</th><th>Metric</th><th>Measured</th>'
            '<th>Operator</th><th>Limit</th><th>Status</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )

    def _analysis_section(self, report: ReportBundle, options: HTMLReportOptions) -> str:
        """Return analysis section.
        
        Args:
            report: Report used by this function.
            options: Options used by this function.
        
        Returns:
            Requested text value.
        """
        chunks = ['<section id="analyses"><h2>Analysis Results</h2>']
        if not report.workflow.analyses:
            chunks.append("<p>No analysis results are available.</p>")
        for analysis in report.workflow.analyses:
            chunks.append('<article class="analysis-card">')
            chunks.append(
                '<div class="analysis-heading">'
                f'<div><h3>{self._e(analysis.tool_name)}</h3>'
                f'<div class="muted">{self._e(analysis.tool_id)} · v{self._e(analysis.tool_version)}</div></div>'
                f'<div><span class="badge badge-{self._status_class(analysis.execution_status)}">{self._e(analysis.execution_status)}</span> '
                f'<span class="badge badge-{self._status_class(analysis.qualification_status)}">{self._e(analysis.qualification_status)}</span></div>'
                "</div>"
            )
            metric_rows = []
            for metric in analysis.metrics:
                description = (
                    f'<div class="metric-description">{self._e(metric.description)}</div>'
                    if options.include_metric_descriptions and metric.description
                    else ""
                )
                metric_rows.append(
                    "<tr>"
                    f"<td>{self._e(metric.name)}{description}</td>"
                    f"<td>{self._metric(metric.value, metric.unit)}</td>"
                    "</tr>"
                )
            if metric_rows:
                chunks.append(
                    '<table class="metric-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
                    f'<tbody>{"".join(metric_rows)}</tbody></table>'
                )

            if analysis.evidence:
                chunks.append("<h4>Evidence</h4><ul class=\"message-list\">")
                for evidence in analysis.evidence:
                    chunks.append(
                        '<li>'
                        f'<span class="badge badge-neutral">{self._e(evidence.strength)}</span> '
                        f'{self._e(evidence.statement)}'
                        "</li>"
                    )
                chunks.append("</ul>")

            if analysis.plots:
                chunks.append('<div class="plot-grid">')
                for plot in analysis.plots:
                    chunks.append(self._plot_html(plot, options))
                chunks.append("</div>")

            messages = list(analysis.warnings) + list(analysis.messages)
            if messages:
                chunks.append('<div class="callout"><strong>Notes</strong><ul>')
                chunks.extend(f"<li>{self._e(message)}</li>" for message in messages)
                chunks.append("</ul></div>")

            if options.include_analysis_metadata and analysis.metadata:
                chunks.append("<details><summary>Analysis metadata</summary><pre>")
                chunks.append(self._e(self._pretty(analysis.metadata)))
                chunks.append("</pre></details>")

            chunks.append("</article>")
        chunks.append("</section>")
        return "".join(chunks)

    def _calibration_section(self, calibration: dict[str, Any]) -> str:
        """Return calibration section.
        
        Args:
            calibration: Calibration used by this function.
        
        Returns:
            Requested text value.
        """
        fit = calibration.get("fit", {})
        model = fit.get("model", {}) if isinstance(fit, dict) else {}
        evaluation = calibration.get("evaluation", {})
        before = evaluation.get("before", {}) if isinstance(evaluation, dict) else {}
        after = evaluation.get("after", {}) if isinstance(evaluation, dict) else {}
        improvement = evaluation.get("improvement", {}) if isinstance(evaluation, dict) else {}
        split = calibration.get("split", {})

        model_rows = [
            ("Method", model.get("method", "—")),
            ("Gain", self._format_value(model.get("gain"))),
            ("Offset", self._format_value(model.get("offset"))),
            ("Training samples", self._format_value(model.get("training_samples"))),
            ("Training rows", self._format_value(split.get("training_rows"))),
            ("Validation rows", self._format_value(split.get("validation_rows"))),
            ("Evaluation basis", calibration.get("evaluation_basis", "held_out_validation")),
        ]

        metric_names = [
            ("RMSE", "rmse", "rmse_reduction_percent"),
            ("MAE", "mae", "mae_reduction_percent"),
            ("Bias", "bias", "absolute_bias_reduction_percent"),
            ("Maximum absolute error", "max_abs_error", "max_abs_error_reduction_percent"),
            ("Error standard deviation", "error_std", "error_std_reduction_percent"),
            ("R²", "r2", None),
        ]
        metric_rows = []
        for label, key, reduction_key in metric_names:
            change = (
                self._metric(improvement.get(reduction_key), "%")
                if reduction_key
                else self._format_value(improvement.get("r2_delta"))
            )
            metric_rows.append(
                "<tr>"
                f"<td>{self._e(label)}</td>"
                f"<td>{self._format_value(before.get(key))}</td>"
                f"<td>{self._format_value(after.get(key))}</td>"
                f"<td>{change}</td>"
                "</tr>"
            )

        warnings = calibration.get("warnings", []) or []
        warning_html = ""
        if warnings:
            warning_html = (
                '<div class="callout"><strong>Calibration notes</strong><ul>'
                + "".join(f"<li>{self._e(item)}</li>" for item in warnings)
                + "</ul></div>"
            )

        return (
            '<section id="calibration"><h2>Held-out Calibration</h2>'
            '<p>Calibration coefficients were fitted on the training subset and '
            'evaluated on a separate held-out validation subset.</p>'
            + self._key_value_table(model_rows)
            + '<h3>Validation Performance</h3>'
            '<table><thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr></thead>'
            f'<tbody>{"".join(metric_rows)}</tbody></table>{warning_html}</section>'
        )

    def _messages_section(self, report: ReportBundle) -> str:
        """Return messages section.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        items: list[tuple[str, str]] = []
        for message in report.workflow.pipeline_errors:
            items.append(("error", message))
        for message in report.workflow.pipeline_warnings:
            items.append(("warning", message))
        for message in report.workflow.workflow_warnings:
            items.append(("warning", message))
        for message in report.workflow.workflow_messages:
            items.append(("info", message))
        if report.metadata.notes:
            items.append(("note", report.metadata.notes))
        if not items:
            return ""
        lis = "".join(
            '<li>'
            f'<span class="badge badge-{self._status_class(kind)}">{self._e(kind)}</span> '
            f'{self._e(message)}</li>'
            for kind, message in items
        )
        return f'<section id="notes"><h2>Warnings and Notes</h2><ul class="message-list">{lis}</ul></section>'

    def _plot_html(self, plot: Any, options: HTMLReportOptions) -> str:
        """Return plot html.
        
        Args:
            plot: Plot used by this function.
            options: Options used by this function.
        
        Returns:
            Requested text value.
        """
        image_html = ""
        if plot.file_path:
            path = Path(plot.file_path).expanduser()
            if options.embed_plot_images:
                embedded = self._image_data_uri(path, options.maximum_embedded_image_bytes)
                if embedded is not None:
                    image_html = f'<img src="{embedded}" alt="{self._e(plot.title)}">'
            elif path.exists():
                image_html = f'<img src="{self._e(str(path.resolve()))}" alt="{self._e(plot.title)}">'
        description = f"<p>{self._e(plot.description)}</p>" if plot.description else ""
        return (
            '<figure class="plot-card">'
            f"{image_html}<figcaption><strong>{self._e(plot.title)}</strong>{description}</figcaption>"
            "</figure>"
        )

    @staticmethod
    def _image_data_uri(path: Path, maximum_bytes: int) -> str | None:
        """Return image data uri.
        
        Args:
            path: Path to the file or directory.
            maximum_bytes: Maximum bytes accepted.
        
        Returns:
            str | None returned by this function.
        """
        try:
            if not path.is_file() or path.stat().st_size > maximum_bytes:
                return None
            mime, _ = mimetypes.guess_type(path.name)
            if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"}:
                return None
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except OSError:
            return None

    def _footer(self, report: ReportBundle) -> str:
        """Return footer.
        
        Args:
            report: Report used by this function.
        
        Returns:
            Requested text value.
        """
        return (
            '<footer><div>Generated by SensorQA.</div>'
            '<div class="muted">Qualification results reflect configured engineering requirements; '
            'statistical relationships and diagnostic evidence should not be interpreted as causal proof.</div>'
            "</footer>"
        )

    @staticmethod
    def _card(label: str, value: str, positive: bool | None) -> str:
        """Return card.
        
        Args:
            label: Label shown to the user.
            value: Value to process.
            positive: Positive used by this function.
        
        Returns:
            Requested text value.
        """
        klass = "neutral" if positive is None else ("positive" if positive else "negative")
        return (
            f'<div class="summary-card {klass}"><div class="summary-label">{html.escape(label)}</div>'
            f'<div class="summary-value">{html.escape(value)}</div></div>'
        )

    @staticmethod
    def _status_bool(status: str) -> bool | None:
        """Return status bool.
        
        Args:
            status: Status value.
        
        Returns:
            bool | None returned by this function.
        """
        normalized = str(status).lower()
        if normalized in {"pass", "passed", "success", "valid"}:
            return True
        if normalized in {"fail", "failed", "error", "invalid"}:
            return False
        return None

    @staticmethod
    def _status_class(status: str) -> str:
        """Return status class.
        
        Args:
            status: Status value.
        
        Returns:
            Requested text value.
        """
        normalized = str(status).lower().replace(" ", "_")
        if normalized in {"pass", "passed", "success", "valid", "strong"}:
            return "success"
        if normalized in {"fail", "failed", "error", "invalid"}:
            return "error"
        if normalized in {"warning", "moderate"}:
            return "warning"
        return "neutral"

    def _key_value_table(self, rows: list[tuple[str, Any]]) -> str:
        """Calculate key value table.
        
        Args:
            rows: Rows used by this function.
        
        Returns:
            Requested text value.
        """
        return (
            '<table class="kv-table"><tbody>'
            + "".join(
                f"<tr><th>{self._e(key)}</th><td>{self._e(value)}</td></tr>"
                for key, value in rows
            )
            + "</tbody></table>"
        )

    def _metric(self, value: Any, unit: str | None) -> str:
        """Return metric.
        
        Args:
            value: Value to process.
            unit: Engineering unit.
        
        Returns:
            Requested text value.
        """
        formatted = self._format_value(value)
        if formatted == "—" or not unit:
            return self._e(formatted)
        return f"{self._e(formatted)} <span class=\"unit\">{self._e(unit)}</span>"

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value.
        
        Args:
            value: Value to process.
        
        Returns:
            Requested text value.
        """
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if abs(value) >= 1e5 or (0 < abs(value) < 1e-4):
                return f"{value:.6g}"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _number(value: int | float) -> str:
        """Return number.
        
        Args:
            value: Value to process.
        
        Returns:
            Requested text value.
        """
        return f"{value:,}"

    @staticmethod
    def _pretty(value: Any) -> str:
        """Return pretty.
        
        Args:
            value: Value to process.
        
        Returns:
            Requested text value.
        """
        import json
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _e(value: Any) -> str:
        """Return e.
        
        Args:
            value: Value to process.
        
        Returns:
            Requested text value.
        """
        return html.escape(str(value), quote=True)

    @staticmethod
    def _document(body: str, *, title: str) -> str:
        """Return document.
        
        Args:
            body: Body used by this function.
            title: Title shown to the user.
        
        Returns:
            Requested text value.
        """
        css = r"""
:root { color-scheme: light; --ink:#18202a; --muted:#657181; --line:#d8dee7; --panel:#f7f9fb; --accent:#1f5f9c; --success:#16794b; --warning:#9b6200; --error:#b42318; }
* { box-sizing:border-box; }
body { margin:0; background:white; color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:40px; }
.report-header { border-bottom:3px solid var(--accent); padding-bottom:20px; margin-bottom:28px; }
.brand { font-weight:800; color:var(--accent); letter-spacing:.08em; text-transform:uppercase; font-size:13px; }
h1 { font-size:30px; margin:4px 0 8px; } h2 { font-size:21px; margin:32px 0 14px; } h3 { font-size:16px; margin:18px 0 10px; } h4 { margin:16px 0 8px; }
.subtitle { font-size:17px; color:var(--muted); margin-bottom:8px; }
.muted,.unit,.metric-description { color:var(--muted); }.metric-description { font-size:12px; margin-top:3px; font-weight:400; }
.card-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.summary-card { border:1px solid var(--line); border-top:4px solid #8993a0; border-radius:8px; padding:14px; background:var(--panel); }
.summary-card.positive { border-top-color:var(--success); }.summary-card.negative { border-top-color:var(--error); }
.summary-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }.summary-value { font-size:22px; font-weight:700; margin-top:3px; }
table { width:100%; border-collapse:collapse; margin:10px 0 18px; } th,td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; } thead th { background:var(--panel); font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
.kv-table { max-width:720px; }.kv-table th { width:240px; color:var(--muted); font-weight:600; background:none; }
.analysis-card { border:1px solid var(--line); border-radius:10px; padding:18px; margin:14px 0; break-inside:avoid; }
.analysis-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }.analysis-heading h3 { margin:0; }
.badge { display:inline-block; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; background:#eef1f5; color:#46515e; }
.badge-success { background:#e7f5ee; color:var(--success); }.badge-error { background:#fdeceb; color:var(--error); }.badge-warning { background:#fff2d8; color:var(--warning); }.badge-neutral { background:#eef1f5; color:#46515e; }
.message-list { padding-left:20px; }.message-list li { margin:6px 0; }.callout { border-left:4px solid var(--warning); background:#fffaf0; padding:10px 14px; margin:14px 0; }
.plot-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }.plot-card { margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; }.plot-card img { width:100%; display:block; }.plot-card figcaption { padding:10px; }
pre { overflow:auto; background:#f2f4f7; padding:12px; border-radius:6px; font-size:12px; }
footer { border-top:1px solid var(--line); margin-top:38px; padding-top:16px; font-size:12px; }
@media print { main { max-width:none; padding:0; } .analysis-card { break-inside:avoid; } a { color:inherit; text-decoration:none; } }
"""
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{css}</style></head>"
            f"<body><main>{body}</main></body></html>"
        )
