# BI front ends

Two front ends over the same 15 extracts the analytics layer publishes, plus the
hosted HTML dashboard that already exists. All three read the *same* governed
numbers; none of them recompute a KPI.

```
bi/powerbi/    Power BI Project (PBIP) - TMDL semantic model + report
bi/spotfire/   IronPython build script for Spotfire Analyst
```

Both are generated:

```
make analytics      # publish the extracts
make bi             # regenerate both front ends from them
```

## Read this first

**Neither has been opened in the product it targets.** Power BI Desktop and
Spotfire Analyst are both Windows-only and neither was available when these were
written, so they are authored against the documented file formats and APIs but
have never been run. Expect to fix something on first open. Both are plain text
specifically so that fixing them is a diff, not a rebuild, and each ships a
fallback that does not depend on the generated layout being correct.

## Power BI

1. Open `bi/powerbi/FraudAnalytics.pbip` in Power BI Desktop
   (needs *Preview features > Power BI Project (.pbip) save format* enabled).
2. **Home > Transform data > Manage parameters** and set `ExtractsFolder` to your
   `artifacts/analytics/extracts` path. This is the only thing you must change.
3. **Refresh.**

What you get: 15 tables and 61 measures across five pages (25 visuals) - Overview, Daily
trend, Threshold economics, Segments, Data quality.

**If the report will not open**, delete `FraudAnalytics.Report/report.json` and
reopen. The semantic model is the valuable half and loads independently; you then
drag fields onto a blank canvas. If the *model* will not load either, use
`fallback/queries.m` and `fallback/measures.dax` to rebuild it in a blank file -
paste each M block into a blank query and each DAX line into a new measure.

### No relationships, deliberately

The model defines none. The KPI and segment extracts are already aggregated by
the Python layer, so a join between them in Power BI could only ever produce a
number that disagrees with the governed one. `Decision Detail` is the single
row-grain table and stands alone for drill-down.

### Measures read, they do not recompute

Every KPI measure is a lookup:

```dax
Value detection rate =
    CALCULATE(SUM('KPI Overall'[value]),
              'KPI Overall'[metric_key] = "value_detection_rate")
```

That looks lazy and is the point. The figure was computed once, in
`src/fraudplat/analytics/metrics.py`, under test. A DAX re-implementation would be
a second definition of the same metric - which is precisely what the metric layer
exists to prevent. Measures carry the Python `description` and `owner` so the
definition travels with the number into the tool.

Measures are generated from `kpi_catalog.csv`, so adding a metric in Python and
re-running `make bi` adds it here with the right name, format, display folder and
target.

## Spotfire

A `.dxp` is a binary container, so unlike the Power BI project it cannot be
authored as text. Scripting it is the next best thing.

1. Open Spotfire Analyst.
2. **Tools > Development > Script Editor** (enable developer tools under
   *Tools > Options > Application* if the menu is missing).
3. Paste `bi/spotfire/build_fraud_analytics.py`, set `EXTRACTS` at the top, run once.
4. Save the `.dxp`.

It loads the 15 extracts, embeds them so the file opens on a machine that cannot
see the folder, and builds the same five pages. It is safe to re-run: tables are
replaced in place and pages of the same name are rebuilt rather than duplicated.

`custom_expressions.md` gives the Spotfire expression for each of the 19 metrics
over `Decision Detail`. Those are for drill-down into cuts the extracts do not
pre-aggregate. The headline figure should always come from `KPI Overall`.

## Which front end for what

| | Best at | Weakness |
| --- | --- | --- |
| Hosted HTML dashboard | one shareable link, no install, the threshold slider | fixed at build time, no ad-hoc slicing |
| Power BI | scheduled refresh, row-level security, PowerPoint export | Windows authoring, licence per viewer |
| Spotfire | heavy ad-hoc analyst exploration, big detail tables | binary file, expensive, scripting to automate |

All three are disposable. The extracts are the product.
