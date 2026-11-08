#!/usr/bin/env python
"""Generate the Power BI and Spotfire front ends from the published extracts.

    python scripts/build_bi_projects.py

The analytics layer ends at 15 BI extracts. This script turns those extracts
into two ready-to-open front ends:

    bi/powerbi/   a Power BI Project (PBIP) - TMDL semantic model + report
    bi/spotfire/  an IronPython build script for Spotfire Analyst

Why generate rather than hand-build: every KPI measure is emitted from
kpi_catalog.csv, which is itself emitted from fraudplat.analytics.metrics.
A metric added in Python therefore appears in both BI tools on the next run,
with the same definition, owner and format - which is the entire point of
having a governed metric layer. Hand-maintained .pbix and .dxp files drift
from the definitions within a quarter; these cannot.

Neither output is binary, so both diff in review.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Both front ends need to be told where the extracts live. The default is a
# placeholder rather than this machine's real path: the generated projects are
# committed, the repo is public, and an absolute path is a home directory in
# plain text. Pass --embed-path for a local-only build.
PLACEHOLDER_PATH = "C:/path/to/fraud-det-plat/artifacts/analytics/extracts"

# Each extract becomes one standalone table. There are deliberately NO
# relationships: the segment and KPI extracts are pre-aggregated by the
# Python layer, so joining them in the BI tool could only ever produce a
# number that disagrees with the governed one.
TABLES = [
    ("KPI Overall", "kpi_overall"),
    ("KPI Daily", "kpi_daily"),
    ("KPI Catalog", "kpi_catalog"),
    ("Decision Detail", "decision_detail"),
    ("Threshold Frontier", "threshold_frontier"),
    ("Threshold Sensitivity", "threshold_sensitivity"),
    ("Data Quality", "data_quality"),
    ("Operations Summary", "operations_summary"),
    ("Analyst Productivity", "analyst_productivity"),
    ("Cost Assumptions", "cost_assumptions"),
    ("Segment Merchant Category", "segment_merchant_category"),
    ("Segment Merchant Country", "segment_merchant_country"),
    ("Segment Channel", "segment_channel"),
    ("Segment Entry Mode", "segment_entry_mode"),
    ("Segment Tenure Band", "segment_tenure_band"),
]

# Columns whose CSV text must be coerced. Everything else is inferred as text
# and left alone, because a wrong implicit cast is worse than a text column.
DATE_COLS = {"calendar_date", "event_time"}
BOOL_COLS = {
    "is_weekend",
    "blocks_publication",
    "requires_settled_label",
    "creates_friction",
    "is_adverse_action",
    "is_fraud",
    "is_declined",
    "is_false_decline",
    "is_caught_fraud",
    "is_missed_fraud",
}
TEXT_COLS = {
    "metric_key",
    "metric_label",
    "definition",
    "unit",
    "domain",
    "owner",
    "direction",
    "tags",
    "day_name",
    "segment_value",
    "segment_name",
    "check",
    "dimension",
    "severity",
    "detail",
    "transaction_id",
    "card_id",
    "merchant_id",
    "merchant_category",
    "merchant_country",
    "merchant_risk_tier",
    "merchant_volume_band",
    "channel",
    "entry_mode",
    "action",
    "triggered_rules",
    "model_version",
    "tenure_band",
    "iso_week",
    "analyst_id",
    "assumption",
}

FORMAT_BY_UNIT = {
    "rate": "0.00%",
    "bps": "#,0.0",
    "currency": "#,0",
    "count": "#,0",
    "ratio": "0.00",
}

DOMAIN_ORDER = ["detection", "decisioning", "operations", "financial", "model", "volume"]


def det_guid(seed: str) -> str:
    """Stable lineage tags, so regenerating produces a clean diff not a new model."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "fraudplat/bi/" + seed))


def read_catalog(extracts: Path) -> list[dict]:
    with (extracts / "kpi_catalog.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(
        key=lambda r: (
            DOMAIN_ORDER.index(r["domain"]) if r["domain"] in DOMAIN_ORDER else 99,
            r["metric_label"],
        )
    )
    return rows


def header_of(extracts: Path, stem: str) -> list[str]:
    with (extracts / f"{stem}.csv").open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def dax_literal(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


# --------------------------------------------------------------------------
# Power BI: TMDL semantic model
# --------------------------------------------------------------------------


def m_query(stem: str, columns: list[str]) -> str:
    """Power Query for one extract. Types are declared explicitly - letting Power
    BI guess turns 'US' into text on one refresh and an error on the next."""
    types = []
    for c in columns:
        if c in DATE_COLS:
            t = "type datetime" if c == "event_time" else "type date"
        elif c in BOOL_COLS:
            t = "type logical"
        elif c in TEXT_COLS:
            t = "type text"
        else:
            t = "type number"
        types.append(f'{{"{c}", {t}}}')
    typed = ", ".join(types)
    return (
        "let\n"
        f'    Source = Csv.Document(File.Contents(ExtractsFolder & "/{stem}.csv"),\n'
        '        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),\n'
        "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n"
        f"    Typed = Table.TransformColumnTypes(Promoted, {{{typed}}})\n"
        "in\n"
        "    Typed"
    )


def tmdl_column(col: str, table: str) -> str:
    if col in DATE_COLS:
        dtype, fmt = (
            ("dateTime", "General Date") if col == "event_time" else ("dateTime", "Long Date")
        )
        extra = f"\n\t\tformatString: {fmt}"
    elif col in BOOL_COLS:
        dtype, extra = "boolean", ""
    elif col in TEXT_COLS:
        dtype, extra = "string", ""
    else:
        dtype, extra = "double", ""
    summarize = "none" if dtype in ("string", "boolean", "dateTime") else "sum"
    return (
        f"\n\tcolumn {col}\n"
        f"\t\tdataType: {dtype}{extra}\n"
        f"\t\tlineageTag: {det_guid(table + '/' + col)}\n"
        f"\t\tsummarizeBy: {summarize}\n"
        f"\t\tsourceColumn: {col}\n\n"
        f"\t\tannotation SummarizationSetBy = Automatic\n"
    )


def tmdl_table(name: str, stem: str, columns: list[str], measures: str = "") -> str:
    body = f"table '{name}'\n\tlineageTag: {det_guid(name)}\n"
    body += measures
    for c in columns:
        body += tmdl_column(c, name)
    q = m_query(stem, columns)
    indented = "\n".join("\t\t\t\t" + ln for ln in q.splitlines())
    body += (
        f"\n\tpartition '{name}' = m\n"
        f"\t\tmode: import\n"
        f"\t\tsource =\n{indented}\n\n"
        f"\tannotation PBI_ResultType = Table\n"
    )
    return body


def overall_measures(catalog: list[dict]) -> str:
    """One measure per governed metric, reading the published value.

    These do not recompute anything. The number was computed once, in Python,
    under test; the BI layer's job is to display it. A DAX re-implementation
    would be a second definition of the same metric - exactly what the metric
    layer exists to prevent.
    """
    out = ""
    for r in catalog:
        fmt = FORMAT_BY_UNIT.get(r["unit"], "0.00")
        expr = (
            f"CALCULATE(SUM('KPI Overall'[value]), "
            f"'KPI Overall'[metric_key] = {dax_literal(r['metric_key'])})"
        )
        desc = r["definition"].replace("\n", " ")
        out += (
            f"\n\tmeasure '{r['metric_label']}' = {expr}\n"
            f"\t\tformatString: {fmt}\n"
            f"\t\tlineageTag: {det_guid('m/overall/' + r['metric_key'])}\n"
            f"\t\tdisplayFolder: {r['domain'].title()}\n"
            f"\t\tdescription: {desc} (owner: {r['owner']})\n"
        )
        if r.get("target"):
            tgt = float(r["target"])
            out += (
                f"\n\tmeasure '{r['metric_label']} target' = {tgt}\n"
                f"\t\tformatString: {fmt}\n"
                f"\t\tlineageTag: {det_guid('m/target/' + r['metric_key'])}\n"
                f"\t\tdisplayFolder: {r['domain'].title()}\\Targets\n"
            )
            cmp_ = ">=" if r["direction"] == "higher is better" else "<="
            if r["direction"] in ("higher is better", "lower is better"):
                out += (
                    f"\n\tmeasure '{r['metric_label']} status' = "
                    f"IF(ISBLANK([{r['metric_label']}]), BLANK(), "
                    f'IF([{r["metric_label"]}] {cmp_} {tgt}, "On target", "Off target"))\n'
                    f"\t\tlineageTag: {det_guid('m/status/' + r['metric_key'])}\n"
                    f"\t\tdisplayFolder: {r['domain'].title()}\\Targets\n"
                )
    return out


def daily_measures(catalog: list[dict]) -> str:
    out = ""
    for r in catalog:
        fmt = FORMAT_BY_UNIT.get(r["unit"], "0.00")
        expr = (
            f"CALCULATE(SUM('KPI Daily'[value]), "
            f"'KPI Daily'[metric_key] = {dax_literal(r['metric_key'])})"
        )
        out += (
            f"\n\tmeasure '{r['metric_label']} (daily)' = {expr}\n"
            f"\t\tformatString: {fmt}\n"
            f"\t\tlineageTag: {det_guid('m/daily/' + r['metric_key'])}\n"
            f"\t\tdisplayFolder: {r['domain'].title()}\n"
        )
    return out


FRONTIER_MEASURES = """
\tmeasure 'Selected threshold' = SELECTEDVALUE('Threshold Frontier'[decline_threshold])
\t\tformatString: 0.00
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Net benefit' = SUM('Threshold Frontier'[net_benefit])
\t\tformatString: #,0
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Net benefit at production' = CALCULATE(SUM('Threshold Frontier'[net_benefit]), ALL('Threshold Frontier'), 'Threshold Frontier'[decline_threshold] = 0.85)
\t\tformatString: #,0
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point
\t\tdescription: 0.85 is the swept point nearest the production threshold of 0.86.

\tmeasure 'Net benefit vs production' = [Net benefit] - [Net benefit at production]
\t\tformatString: +#,0;-#,0;0
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Frontier detection rate' = SUM('Threshold Frontier'[detection_rate])
\t\tformatString: 0.0%
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Frontier value detection' = SUM('Threshold Frontier'[value_detection_rate])
\t\tformatString: 0.0%
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Frontier precision' = SUM('Threshold Frontier'[precision])
\t\tformatString: 0.0%
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Frontier false declines' = SUM('Threshold Frontier'[false_declines])
\t\tformatString: #,0
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point

\tmeasure 'Frontier alerts per 10k' = SUM('Threshold Frontier'[alerts_per_10k_score_only])
\t\tformatString: #,0.0
\t\tlineageTag: {}
\t\tdisplayFolder: Operating point
\t\tdescription: Score-only alert load. Distinct from the alerts_per_10k KPI, which counts the realised queue including rule escalations.
""".format(*[det_guid(f"m/frontier/{i}") for i in range(9)])

QUALITY_MEASURES = """
\tmeasure 'Checks passing' = CALCULATE(COUNTROWS('Data Quality'), 'Data Quality'[severity] = "pass")
\t\tformatString: #,0
\t\tlineageTag: {}

\tmeasure 'Checks total' = COUNTROWS('Data Quality')
\t\tformatString: #,0
\t\tlineageTag: {}

\tmeasure 'Publishable' = IF(CALCULATE(COUNTROWS('Data Quality'), 'Data Quality'[severity] <> "pass", 'Data Quality'[blocks_publication] = TRUE()) > 0, "WITHHELD", "PUBLISHABLE")
\t\tlineageTag: {}
\t\tdescription: Mirrors QualityReport.publishable - a non-blocking warning does not withhold the refresh.
""".format(*[det_guid(f"m/quality/{i}") for i in range(3)])


# --------------------------------------------------------------------------
# Power BI: report layer
# --------------------------------------------------------------------------


def fld(kind: str, alias: str, entity: str, prop: str) -> dict:
    return {
        kind: {"Expression": {"SourceRef": {"Source": alias}}, "Property": prop},
        "Name": f"{entity}.{prop}",
        "NativeReferenceName": prop,
    }


def visual(name, vtype, x, y, w, h, entity, select, roles, title=None, where=None):
    """One visualContainer. Power BI stores each visual's config as a JSON string
    inside the report JSON, which is why this is built as a dict and dumped."""
    proj = {role: [{"queryRef": select[i]["Name"]} for i in idxs] for role, idxs in roles.items()}
    single = {
        "visualType": vtype,
        "projections": proj,
        "prototypeQuery": {
            "Version": 2,
            "From": [{"Name": "t", "Entity": entity, "Type": 0}],
            "Select": select,
        },
        "drillFilterOtherVisuals": True,
        "objects": {},
        "vcObjects": {},
    }
    if where:
        single["prototypeQuery"]["Where"] = where
    if title:
        single["vcObjects"]["title"] = [
            {
                "properties": {
                    "show": {"expr": {"Literal": {"Value": "true"}}},
                    "text": {"expr": {"Literal": {"Value": "'" + title.replace("'", "''") + "'"}}},
                }
            }
        ]
    cfg = {
        "name": name,
        "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": 0, "width": w, "height": h}}],
        "singleVisual": single,
    }
    return {
        "x": x,
        "y": y,
        "z": 0,
        "width": w,
        "height": h,
        "config": json.dumps(cfg, separators=(",", ":")),
        "filters": "[]",
    }


def in_filter(alias: str, entity: str, prop: str, values: list[str]) -> list:
    return [
        {
            "Condition": {
                "In": {
                    "Expressions": [
                        {
                            "Column": {
                                "Expression": {"SourceRef": {"Source": alias}},
                                "Property": prop,
                            }
                        }
                    ],
                    "Values": [[{"Literal": {"Value": "'" + v + "'"}}] for v in values],
                }
            }
        }
    ]


def page(name, display, ordinal, visuals):
    return {
        "name": name,
        "displayName": display,
        "filters": "[]",
        "ordinal": ordinal,
        "visualContainers": visuals,
        "config": "{}",
        "displayOption": 1,
        "width": 1280,
        "height": 720,
    }


def build_report(catalog: list[dict]) -> dict:
    by_key = {r["metric_key"]: r["metric_label"] for r in catalog}
    KO, KD, TF, DQ, SC = (
        "KPI Overall",
        "KPI Daily",
        "Threshold Frontier",
        "Data Quality",
        "Segment Merchant Category",
    )
    pages = []

    # --- page 1: overview -------------------------------------------------
    v = []
    cards = ["value_detection_rate", "false_positive_ratio", "fraud_loss_bps", "detection_rate"]
    for i, k in enumerate(cards):
        sel = [fld("Measure", "t", KO, by_key[k])]
        v.append(
            visual(
                f"ov_card{i}",
                "card",
                16 + i * 312,
                16,
                296,
                120,
                KO,
                sel,
                {"Values": [0]},
                title=by_key[k],
            )
        )
    sel = [fld("Measure", "t", DQ, "Checks passing"), fld("Measure", "t", DQ, "Publishable")]
    v.append(
        visual(
            "ov_q",
            "multiRowCard",
            16,
            152,
            296,
            150,
            DQ,
            sel,
            {"Values": [0, 1]},
            title="Data quality gate",
        )
    )
    sel = [
        fld("Column", "t", KO, "domain"),
        fld("Column", "t", KO, "metric_label"),
        fld("Column", "t", KO, "value"),
        fld("Column", "t", KO, "owner"),
    ]
    v.append(
        visual(
            "ov_tbl",
            "tableEx",
            328,
            152,
            936,
            552,
            KO,
            sel,
            {"Values": [0, 1, 2, 3]},
            title="Metric layer - all 19 governed definitions",
        )
    )
    sel = [fld("Column", "t", KO, "domain"), fld("Column", "t", KO, "metric_key")]
    v.append(
        visual("ov_slicer", "slicer", 16, 318, 296, 386, KO, sel, {"Values": [0]}, title="Domain")
    )
    pages.append(page("pgOverview", "Overview", 0, v))

    # --- page 2: daily trend ----------------------------------------------
    v = []
    trends = [
        ("Detection", ["detection_rate", "value_detection_rate"], 16, 16),
        ("Fraud prevalence", ["fraud_prevalence_bps"], 648, 16),
        ("Model health", ["mean_risk_score", "high_risk_rate"], 16, 364),
        ("Decision mix", ["decline_rate", "challenge_rate", "review_rate"], 648, 364),
    ]
    for i, (title, keys, x, y) in enumerate(trends):
        sel = [
            fld("Column", "t", KD, "calendar_date"),
            fld("Measure", "t", KD, "value"),
            fld("Column", "t", KD, "metric_key"),
        ]
        sel[1] = {
            "Aggregation": {
                "Expression": {
                    "Column": {"Expression": {"SourceRef": {"Source": "t"}}, "Property": "value"}
                },
                "Function": 0,
            },
            "Name": "Sum(KPI Daily.value)",
            "NativeReferenceName": "value",
        }
        v.append(
            visual(
                f"dt{i}",
                "lineChart",
                x,
                y,
                616,
                332,
                KD,
                sel,
                {"Category": [0], "Y": [1], "Series": [2]},
                title=title,
                where=in_filter("t", KD, "metric_key", keys),
            )
        )
    pages.append(page("pgDaily", "Daily trend", 1, v))

    # --- page 3: threshold economics --------------------------------------
    v = []
    sel = [fld("Column", "t", TF, "decline_threshold")]
    v.append(
        visual(
            "th_slicer",
            "slicer",
            16,
            16,
            296,
            200,
            TF,
            sel,
            {"Values": [0]},
            title="Decline threshold",
        )
    )
    for i, (m, lbl) in enumerate(
        [
            ("Net benefit", "Net benefit"),
            ("Net benefit vs production", "vs production (0.86)"),
            ("Frontier value detection", "Value detection"),
            ("Frontier precision", "Decline precision"),
            ("Frontier false declines", "False declines"),
            ("Frontier alerts per 10k", "Alerts per 10k"),
        ]
    ):
        sel = [fld("Measure", "t", TF, m)]
        v.append(
            visual(
                f"th_c{i}",
                "card",
                16 + (i % 2) * 152,
                232 + (i // 2) * 120,
                144,
                112,
                TF,
                sel,
                {"Values": [0]},
                title=lbl,
            )
        )
    sel = [
        fld("Column", "t", TF, "decline_threshold"),
        {
            "Aggregation": {
                "Expression": {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "t"}},
                        "Property": "net_benefit",
                    }
                },
                "Function": 0,
            },
            "Name": "Sum(Threshold Frontier.net_benefit)",
            "NativeReferenceName": "net_benefit",
        },
    ]
    v.append(
        visual(
            "th_line",
            "lineChart",
            328,
            16,
            936,
            400,
            TF,
            sel,
            {"Category": [0], "Y": [1]},
            title="Net benefit across all 95 candidate thresholds",
        )
    )
    sel = [
        fld("Column", "t", TF, c)
        for c in (
            "decline_threshold",
            "detection_rate",
            "value_detection_rate",
            "precision",
            "false_declines",
            "alerts_per_10k_score_only",
            "net_benefit",
        )
    ]
    v.append(
        visual(
            "th_tbl",
            "tableEx",
            328,
            432,
            936,
            272,
            TF,
            sel,
            {"Values": list(range(7))},
            title="Frontier detail",
        )
    )
    pages.append(page("pgThreshold", "Threshold economics", 2, v))

    # --- page 4: segments -------------------------------------------------
    v = []
    for i, (col, title, x) in enumerate(
        [
            ("fraud_prevalence_bps", "Fraud prevalence (bps) - where fraud concentrates", 16),
            ("value_detection_rate", "Value detection rate - how much of it we stop", 648),
        ]
    ):
        sel = [
            fld("Column", "t", SC, "segment_value"),
            {
                "Aggregation": {
                    "Expression": {
                        "Column": {"Expression": {"SourceRef": {"Source": "t"}}, "Property": col}
                    },
                    "Function": 0,
                },
                "Name": f"Sum({SC}.{col})",
                "NativeReferenceName": col,
            },
        ]
        v.append(
            visual(
                f"sg{i}",
                "barChart",
                x,
                16,
                616,
                420,
                SC,
                sel,
                {"Category": [0], "Y": [1]},
                title=title,
            )
        )
    sel = [
        fld("Column", "t", SC, c)
        for c in (
            "segment_value",
            "authorizations",
            "attempted_spend",
            "fraud_prevalence_bps",
            "detection_rate",
            "value_detection_rate",
            "false_decline_rate",
        )
    ]
    v.append(
        visual(
            "sg_tbl",
            "tableEx",
            16,
            452,
            1248,
            252,
            SC,
            sel,
            {"Values": list(range(7))},
            title="Merchant category detail",
        )
    )
    pages.append(page("pgSegments", "Segments", 3, v))

    # --- page 5: data quality ---------------------------------------------
    v = []
    sel = [
        fld("Column", "t", DQ, c)
        for c in (
            "dimension",
            "check",
            "severity",
            "measured",
            "threshold",
            "blocks_publication",
            "detail",
        )
    ]
    v.append(
        visual(
            "dq_tbl",
            "tableEx",
            16,
            16,
            1248,
            560,
            DQ,
            sel,
            {"Values": list(range(7))},
            title="17 checks - a blocking failure aborts the refresh",
        )
    )
    sel = [
        fld("Measure", "t", DQ, "Checks passing"),
        fld("Measure", "t", DQ, "Checks total"),
        fld("Measure", "t", DQ, "Publishable"),
    ]
    v.append(
        visual(
            "dq_card",
            "multiRowCard",
            16,
            592,
            1248,
            112,
            DQ,
            sel,
            {"Values": [0, 1, 2]},
            title="Gate status",
        )
    )
    pages.append(page("pgQuality", "Data quality", 4, v))

    return {
        "config": json.dumps(
            {"version": "5.43", "activeSectionIndex": 0, "defaultDrillFilterOtherVisuals": True},
            separators=(",", ":"),
        ),
        "layoutOptimization": 0,
        "resourcePackages": [],
        "pods": [],
        "publicCustomVisuals": [],
        "sections": pages,
        "filters": "[]",
    }


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def write_powerbi(out: Path, extracts: Path, folder: str, catalog: list[dict]) -> int:
    model = out / "FraudAnalytics.SemanticModel"
    report = out / "FraudAnalytics.Report"
    defs = model / "definition"
    (defs / "tables").mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    (out / "FraudAnalytics.pbip").write_text(
        json.dumps(
            {
                "version": "1.0",
                "artifacts": [{"report": {"path": "FraudAnalytics.Report"}}],
                "settings": {"enableAutoRecovery": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (model / "definition.pbism").write_text(
        json.dumps({"version": "4.2", "settings": {}}, indent=2) + "\n", encoding="utf-8"
    )
    (defs / "database.tmdl").write_text("database\n\tcompatibilityLevel: 1567\n", encoding="utf-8")

    # The one thing a new user must change: where the extracts live.
    (defs / "expressions.tmdl").write_text(
        'expression ExtractsFolder = "{}" meta [IsParameterQuery=true, '
        'Type="Text", IsParameterQueryRequired=true]\n'
        "\tlineageTag: {}\n\n"
        "\tannotation PBI_NavigationStepName = Navigation\n\n"
        "\tannotation PBI_ResultType = Text\n".format(
            folder.replace("\\", "/"), det_guid("expr/ExtractsFolder")
        ),
        encoding="utf-8",
    )

    written = 0
    for name, stem in TABLES:
        cols = header_of(extracts, stem)
        measures = ""
        if name == "KPI Overall":
            measures = overall_measures(catalog)
        elif name == "KPI Daily":
            measures = daily_measures(catalog)
        elif name == "Threshold Frontier":
            measures = FRONTIER_MEASURES
        elif name == "Data Quality":
            measures = QUALITY_MEASURES
        (defs / "tables" / f"{name}.tmdl").write_text(
            tmdl_table(name, stem, cols, measures), encoding="utf-8"
        )
        written += 1

    refs = "\n".join(f"ref table '{n}'" for n, _ in TABLES)
    (defs / "model.tmdl").write_text(
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tsourceQueryCulture: en-GB\n"
        "\tdataAccessOptions\n"
        "\t\tlegacyRedirects\n"
        "\t\treturnErrorValuesAsNull\n\n"
        f"annotation PBI_QueryOrder = {json.dumps([n for n, _ in TABLES])}\n\n"
        f"ref expression ExtractsFolder\n\n{refs}\n",
        encoding="utf-8",
    )

    (report / "definition.pbir").write_text(
        json.dumps(
            {
                "version": "4.0",
                "datasetReference": {"byPath": {"path": "../FraudAnalytics.SemanticModel"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (report / "report.json").write_text(
        json.dumps(build_report(catalog), indent=2) + "\n", encoding="utf-8"
    )

    # Fallback: if the generated report will not open, these two files rebuild
    # it by hand in a blank .pbix in about twenty minutes.
    fb = out / "fallback"
    fb.mkdir(exist_ok=True)
    qs = [
        "// Paste each block into Home > Transform data > New Source > Blank Query >",
        "// Advanced Editor. Create ExtractsFolder as a Text parameter first.",
        "",
        f'// ExtractsFolder (Text parameter) = "{folder.replace(chr(92), "/")}"',
        "",
    ]
    for name, stem in TABLES:
        qs += [f"// ---- {name} ----", m_query(stem, header_of(extracts, stem)), ""]
    (fb / "queries.m").write_text("\n".join(qs), encoding="utf-8")

    dx = [
        "// One measure per governed metric, generated from kpi_catalog.csv.",
        "// These read the published value - they do not recompute it. The number",
        "// was computed once in Python under test; re-deriving it in DAX would",
        "// create a second definition of the same metric.",
        "",
    ]
    for r in catalog:
        dx += [
            f"// {r['metric_label']} - {r['definition']} (owner: {r['owner']})",
            f"{r['metric_label']} = CALCULATE(SUM('KPI Overall'[value]), "
            f"'KPI Overall'[metric_key] = {dax_literal(r['metric_key'])})",
            "",
        ]
    (fb / "measures.dax").write_text("\n".join(dx), encoding="utf-8")
    return written


SPOTFIRE_HEAD = '''# IronPython build script for Spotfire Analyst.
#
#   Tools > Development > Script Editor  (or Edit > Document Properties >
#   Script tab > New), paste this in, set EXTRACTS below, and run once.
#
# Spotfire's .dxp is a binary container, so an analysis cannot be authored as
# text the way the Power BI project can. Scripting it is the next best thing:
# this file IS the dashboard definition, it diffs in review, and re-running it
# against a refreshed extract folder rebuilds every page identically.
#
# Generated by scripts/build_bi_projects.py - edit that, not this.

import os
from Spotfire.Dxp.Data import DataTableSaveSettings
from Spotfire.Dxp.Data.Import import TextDataReaderSettings, TextFileDataSource
from Spotfire.Dxp.Application.Visuals import (
    BarChart, BarChartOrientation, HtmlTextArea, LineChart, TablePlot,
)

# ---------------------------------------------------------------- settings
EXTRACTS = r"{extracts}"

TABLES = {tables}


def load(name, stem):
    """Load one extract, replacing it in place if the table already exists so
    the script is safe to re-run against a refreshed folder."""
    path = os.path.join(EXTRACTS, stem + ".csv")
    if not os.path.exists(path):
        print("MISSING: " + path)
        return None
    settings = TextDataReaderSettings()
    settings.Separator = ","
    settings.AddColumnNameRow(0)
    source = TextFileDataSource(path, settings)
    if Document.Data.Tables.Contains(name):
        Document.Data.Tables[name].ReplaceData(source)
    else:
        Document.Data.Tables.Add(name, source)
    table = Document.Data.Tables[name]
    # Embed the data so the .dxp opens on a machine that cannot see the folder.
    table.SaveSettings = DataTableSaveSettings(table, False, True)
    return table


def fresh_page(title):
    """Replace a page of the same name rather than stacking duplicates."""
    for existing in list(Document.Pages):
        if existing.Title == title:
            Document.Pages.Remove(existing)
    return Document.Pages.AddNew(title)


def text(page, html):
    area = page.Visuals.AddNew[HtmlTextArea]()
    area.HtmlContent = html
    return area


def line(page, table, title, x_expr, y_expr, color_expr=None, where=None):
    visual = page.Visuals.AddNew[LineChart]()
    visual.Title = title
    visual.Data.DataTableReference = table
    if where:
        visual.Data.WhereClauseExpression = where
    visual.XAxis.Expression = x_expr
    visual.YAxis.Expression = y_expr
    if color_expr:
        visual.ColorAxis.Expression = color_expr
    return visual


def bars(page, table, title, x_expr, y_expr, horizontal=True, color_expr=None, where=None):
    visual = page.Visuals.AddNew[BarChart]()
    visual.Title = title
    visual.Data.DataTableReference = table
    if where:
        visual.Data.WhereClauseExpression = where
    visual.XAxis.Expression = x_expr
    visual.YAxis.Expression = y_expr
    if color_expr:
        visual.ColorAxis.Expression = color_expr
    if horizontal:
        visual.Orientation = BarChartOrientation.Horizontal
    visual.SortedBars = True
    return visual


def grid(page, table, title, columns=None):
    visual = page.Visuals.AddNew[TablePlot]()
    visual.Title = title
    visual.Data.DataTableReference = table
    if columns:
        visual.Columns.Clear()
        for column in columns:
            visual.Columns.Add(table.Columns[column])
    return visual
'''

SPOTFIRE_BODY = '''

# ------------------------------------------------------------------ load
loaded = {{}}
for display_name, stem in TABLES:
    loaded[display_name] = load(display_name, stem)

kpi_overall = loaded["KPI Overall"]
kpi_daily = loaded["KPI Daily"]
frontier = loaded["Threshold Frontier"]
quality = loaded["Data Quality"]
category = loaded["Segment Merchant Category"]

# --------------------------------------------------------------- page 1
page1 = fresh_page("Overview")
text(page1, """{overview_html}""")
grid(page1, kpi_overall, "Metric layer - 19 governed definitions",
     ["domain", "metric_label", "value", "unit", "owner", "direction", "target"])

# --------------------------------------------------------------- page 2
page2 = fresh_page("Daily trend")
line(page2, kpi_daily, "Detection - count recall vs value recall",
     "[calendar_date]", "Sum([value])", "[metric_key]",
     "[metric_key] in ('detection_rate', 'value_detection_rate')")
line(page2, kpi_daily, "Fraud prevalence (bps)",
     "[calendar_date]", "Sum([value])", None,
     "[metric_key] = 'fraud_prevalence_bps'")
line(page2, kpi_daily, "Model health - leading drift signals",
     "[calendar_date]", "Sum([value])", "[metric_key]",
     "[metric_key] in ('mean_risk_score', 'high_risk_rate')")

# --------------------------------------------------------------- page 3
page3 = fresh_page("Threshold economics")
text(page3, """{threshold_html}""")
line(page3, frontier, "Net benefit by decline threshold",
     "[decline_threshold]", "Sum([net_benefit])")
line(page3, frontier, "What it costs - detection against false declines",
     "[decline_threshold]", "Sum([detection_rate])")
grid(page3, frontier, "Frontier detail",
     ["decline_threshold", "detection_rate", "value_detection_rate", "precision",
      "false_declines", "alerts_per_10k_score_only", "net_benefit"])

# --------------------------------------------------------------- page 4
page4 = fresh_page("Segments")
bars(page4, category, "Fraud prevalence (bps) - where fraud concentrates",
     "[segment_value]", "Sum([fraud_prevalence_bps])")
bars(page4, category, "Value detection rate - how much of it we stop",
     "[segment_value]", "Sum([value_detection_rate])")
grid(page4, category, "Merchant category detail")

# --------------------------------------------------------------- page 5
page5 = fresh_page("Data quality")
text(page5, """{quality_html}""")
bars(page5, quality, "Checks by dimension", "[dimension]", "Count()", False, "[severity]")
grid(page5, quality, "17 checks",
     ["dimension", "check", "severity", "measured", "threshold", "blocks_publication", "detail"])

Document.Pages.SetActivePage(page1)
print("Built 5 pages from " + EXTRACTS)
'''


def write_spotfire(out: Path, folder: str, catalog: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    k = summary.get("kpis", {})
    overview = (
        "<h2>Fraud operating point</h2>"
        "<p>Holdout window {w0} to {w1} &middot; {rows:,} authorizations the model never "
        "trained on &middot; model {mv}.</p>"
        "<p><b>{vdr:.1%}</b> of confirmed-fraud value stopped at a decline precision of "
        "<b>{prec:.1%}</b>. Figures are produced by the Python analytics layer and read "
        "here, not recomputed - if a Spotfire expression disagrees with one of these, "
        "the Python is right.</p>"
        "<p style='color:#777'>Synthetic data. Analyst activity is simulated.</p>"
    ).format(
        w0=summary.get("window", ["?", "?"])[0],
        w1=summary.get("window", ["?", "?"])[1],
        rows=int(summary.get("rows", 0)),
        mv=summary.get("model_version", "?"),
        vdr=k.get("value_detection_rate", 0),
        prec=k.get("precision_at_decline", 0),
    )
    threshold = (
        "<h3>Threshold economics</h3>"
        "<p>Production declines at <b>{cur:.2f}</b>. Net benefit peaks at <b>{rec:.2f}</b>, "
        "but that point pushes the review queue past analyst capacity, so the recommended "
        "move is <b>{con:.2f}</b>.</p>"
        "<p>Filter <i>decline_threshold</i> in the filter panel to read any single point.</p>"
    ).format(
        cur=summary.get("current_threshold", 0),
        rec=summary.get("recommended_threshold", 0),
        con=summary.get("constrained_threshold", 0),
    )
    qual = (
        "<h3>Data quality gate</h3>"
        "<p>This is a gate, not a report. A failing check marked "
        "<i>blocks_publication</i> aborts the refresh and leaves the previous extracts "
        "in place - stale but correct beats fresh but wrong when the output goes into "
        "a client pack.</p>"
    )
    script = SPOTFIRE_HEAD.format(
        extracts=folder,
        tables=json.dumps([[n, s] for n, s in TABLES], indent=4),
    ) + SPOTFIRE_BODY.format(overview_html=overview, threshold_html=threshold, quality_html=qual)
    (out / "build_fraud_analytics.py").write_text(script, encoding="utf-8")

    lines = [
        "# Spotfire custom expressions",
        "",
        "Every governed metric, as the Spotfire expression that reproduces it from",
        "`Decision Detail`. **Use these for drill-down only.** The headline figure",
        "should always come from the `KPI Overall` table, which carries the value the",
        "Python layer computed under test. These expressions exist so an analyst can",
        "break a number down by a dimension the extracts do not pre-aggregate - not so",
        "the number can be re-derived.",
        "",
        "| Metric | Owner | Spotfire expression over `Decision Detail` |",
        "| --- | --- | --- |",
    ]
    expr = {
        "authorizations": "Count()",
        "authorized_spend": "Sum(If([action]<>'decline',[amount],0))",
        "attempted_spend": "Sum([amount])",
        "decline_rate": "Sum(If([is_declined],1,0)) / Count()",
        "challenge_rate": "Sum(If([action]='step_up',1,0)) / Count()",
        "review_rate": "Sum(If([action]='review',1,0)) / Count()",
        "alerts_per_10k": "10000 * Sum(If([action]='review',1,0)) / Count()",
        "rule_override_rate": "Sum(If([triggered_rule_count]>0,1,0)) / Count()",
        "fraud_prevalence_bps": "10000 * Sum(If([is_fraud],1,0)) / Count()",
        "detection_rate": "Sum(If([is_caught_fraud],1,0)) / Sum(If([is_fraud],1,0))",
        "value_detection_rate": "Sum([fraud_value_stopped]) / (Sum([fraud_value_stopped])+Sum([fraud_value_lost]))",
        "precision_at_decline": "Sum(If([is_caught_fraud],1,0)) / Sum(If([is_declined],1,0))",
        "false_decline_rate": "Sum(If([is_false_decline],1,0)) / Sum(If([is_fraud],0,1))",
        "false_positive_ratio": "Sum(If([is_false_decline],1,0)) / Sum(If([is_caught_fraud],1,0))",
        "fraud_loss_bps": "10000 * Sum([fraud_value_lost]) / Sum([amount])",
        "mean_risk_score": "Avg([risk_score])",
        "high_risk_rate": "Sum(If([risk_score]>0.55,1,0)) / Count()",
        "cold_start_rate": "Sum(If([sequence_score]=0.5,1,0)) / Count()",
        "label_coverage": "Sum(If([is_fraud] is null,0,1)) / Count()",
    }
    for r in catalog:
        e = expr.get(r["metric_key"], "-")
        lines.append(f"| {r['metric_label']} | {r['owner']} | `{e}` |")
    lines += [
        "",
        "## Why these are not the source of truth",
        "",
        "`value_detection_rate` above divides stopped value by total fraud value. The",
        "Python definition does the same thing - but it also decides what counts as",
        "settled, which rows are in the holdout, and how a null label is treated. Those",
        "decisions are tested; a cell in a Spotfire expression editor is not. Where the",
        "two disagree, the disagreement is the bug, and the fix belongs in",
        "`src/fraudplat/analytics/metrics.py`.",
        "",
    ]
    (out / "custom_expressions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracts", type=Path, default=ROOT / "artifacts" / "analytics" / "extracts")
    ap.add_argument("--out", type=Path, default=ROOT / "bi")
    ap.add_argument("--clean", action="store_true", help="remove the output tree first")
    ap.add_argument(
        "--embed-path",
        action="store_true",
        help="bake this machine's absolute extracts path into the projects. Off by "
        "default: the repo is public, and an absolute path leaks a home directory. "
        "Without it both front ends ship a placeholder the user sets once.",
    )
    args = ap.parse_args()

    if not args.extracts.exists():
        print(f"no extracts at {args.extracts}; run scripts/build_analytics.py first")
        return 1
    missing = [s for _, s in TABLES if not (args.extracts / f"{s}.csv").exists()]
    if missing:
        print("missing extracts: " + ", ".join(missing))
        return 1

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    catalog = read_catalog(args.extracts)
    summary_path = args.extracts.parent / "analytics_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    folder = str(args.extracts.resolve()) if args.embed_path else PLACEHOLDER_PATH
    n = write_powerbi(args.out / "powerbi", args.extracts, folder, catalog)
    print(f"power bi : {n} tables, {len(catalog)} metrics -> {args.out / 'powerbi'}")

    write_spotfire(args.out / "spotfire", folder, catalog, summary)
    print(f"spotfire : build script + {len(catalog)} expressions -> {args.out / 'spotfire'}")
    if not args.embed_path:
        print(f"           extracts path left as {PLACEHOLDER_PATH} - set it on first open")
    return 0


if __name__ == "__main__":
    sys.exit(main())
