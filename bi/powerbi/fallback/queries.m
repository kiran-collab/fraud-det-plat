// Paste each block into Home > Transform data > New Source > Blank Query >
// Advanced Editor. Create ExtractsFolder as a Text parameter first.

// ExtractsFolder (Text parameter) = "C:/path/to/fraud-det-plat/artifacts/analytics/extracts"

// ---- KPI Overall ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/kpi_overall.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"metric_key", type text}, {"value", type number}, {"metric_label", type text}, {"definition", type text}, {"unit", type text}, {"domain", type text}, {"owner", type text}, {"direction", type text}, {"requires_settled_label", type logical}, {"target", type number}, {"tags", type text}})
in
    Typed

// ---- KPI Daily ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/kpi_daily.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"calendar_date", type date}, {"day_name", type text}, {"is_weekend", type logical}, {"metric_key", type text}, {"value", type number}})
in
    Typed

// ---- KPI Catalog ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/kpi_catalog.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"metric_key", type text}, {"metric_label", type text}, {"definition", type text}, {"unit", type text}, {"domain", type text}, {"owner", type text}, {"direction", type text}, {"requires_settled_label", type logical}, {"target", type number}, {"tags", type text}})
in
    Typed

// ---- Decision Detail ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/decision_detail.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"decision_sk", type number}, {"transaction_id", type text}, {"event_time", type datetime}, {"calendar_date", type date}, {"iso_week", type text}, {"day_name", type text}, {"hour_of_day", type number}, {"card_id", type text}, {"tenure_band", type text}, {"lifetime_decisions", type number}, {"merchant_id", type text}, {"merchant_category", type text}, {"merchant_country", type text}, {"merchant_risk_tier", type text}, {"merchant_volume_band", type text}, {"channel", type text}, {"entry_mode", type text}, {"amount", type number}, {"risk_score", type number}, {"supervised_score", type number}, {"anomaly_score", type number}, {"sequence_score", type number}, {"action", type text}, {"action_rank", type number}, {"creates_friction", type logical}, {"is_adverse_action", type logical}, {"triggered_rule_count", type number}, {"triggered_rules", type text}, {"is_fraud", type logical}, {"is_declined", type logical}, {"is_false_decline", type logical}, {"is_caught_fraud", type logical}, {"is_missed_fraud", type logical}, {"fraud_value_stopped", type number}, {"fraud_value_lost", type number}, {"model_version", type text}})
in
    Typed

// ---- Threshold Frontier ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/threshold_frontier.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"decline_threshold", type number}, {"review_threshold", type number}, {"declines", type number}, {"reviews", type number}, {"caught_fraud", type number}, {"missed_fraud", type number}, {"false_declines", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision", type number}, {"false_decline_rate", type number}, {"alerts_per_10k_score_only", type number}, {"fraud_value_stopped", type number}, {"fraud_value_lost", type number}, {"gross_benefit", type number}, {"false_decline_cost", type number}, {"review_cost", type number}, {"net_benefit", type number}})
in
    Typed

// ---- Threshold Sensitivity ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/threshold_sensitivity.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"assumption", type text}, {"multiplier", type number}, {"value", type number}, {"optimal_threshold", type number}, {"net_benefit", type number}, {"detection_rate", type number}, {"false_declines", type number}})
in
    Typed

// ---- Data Quality ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/data_quality.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"check", type text}, {"dimension", type text}, {"measured", type number}, {"threshold", type number}, {"severity", type text}, {"blocks_publication", type logical}, {"detail", type text}})
in
    Typed

// ---- Operations Summary ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/operations_summary.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"alerts", type number}, {"analysts_active", type number}, {"avg_handling_minutes", type number}, {"median_handling_minutes", type number}, {"p90_handling_minutes", type number}, {"total_analyst_hours", type number}, {"sla_breach_rate", type number}, {"open_rate", type number}, {"case_conversion_rate", type number}, {"analyst_agreement_rate", type number}, {"review_cost_total", type number}, {"value_confirmed_fraud", type number}})
in
    Typed

// ---- Analyst Productivity ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/analyst_productivity.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"analyst_id", type text}, {"alerts_worked", type number}, {"avg_handling_minutes", type number}, {"sla_breach_rate", type number}, {"fraud_confirmed", type number}})
in
    Typed

// ---- Cost Assumptions ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/cost_assumptions.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"recovery_rate", type number}, {"review_cost", type number}, {"false_decline_cost", type number}, {"step_up_cost", type number}, {"chargeback_fee", type number}})
in
    Typed

// ---- Segment Merchant Category ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/segment_merchant_category.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment_value", type text}, {"segment_name", type text}, {"authorizations", type number}, {"authorized_spend", type number}, {"attempted_spend", type number}, {"decline_rate", type number}, {"challenge_rate", type number}, {"review_rate", type number}, {"alerts_per_10k", type number}, {"rule_override_rate", type number}, {"fraud_prevalence_bps", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision_at_decline", type number}, {"false_decline_rate", type number}, {"false_positive_ratio", type number}, {"fraud_loss_bps", type number}, {"mean_risk_score", type number}, {"high_risk_rate", type number}, {"cold_start_rate", type number}, {"label_coverage", type number}})
in
    Typed

// ---- Segment Merchant Country ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/segment_merchant_country.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment_value", type text}, {"segment_name", type text}, {"authorizations", type number}, {"authorized_spend", type number}, {"attempted_spend", type number}, {"decline_rate", type number}, {"challenge_rate", type number}, {"review_rate", type number}, {"alerts_per_10k", type number}, {"rule_override_rate", type number}, {"fraud_prevalence_bps", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision_at_decline", type number}, {"false_decline_rate", type number}, {"false_positive_ratio", type number}, {"fraud_loss_bps", type number}, {"mean_risk_score", type number}, {"high_risk_rate", type number}, {"cold_start_rate", type number}, {"label_coverage", type number}})
in
    Typed

// ---- Segment Channel ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/segment_channel.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment_value", type text}, {"segment_name", type text}, {"authorizations", type number}, {"authorized_spend", type number}, {"attempted_spend", type number}, {"decline_rate", type number}, {"challenge_rate", type number}, {"review_rate", type number}, {"alerts_per_10k", type number}, {"rule_override_rate", type number}, {"fraud_prevalence_bps", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision_at_decline", type number}, {"false_decline_rate", type number}, {"false_positive_ratio", type number}, {"fraud_loss_bps", type number}, {"mean_risk_score", type number}, {"high_risk_rate", type number}, {"cold_start_rate", type number}, {"label_coverage", type number}})
in
    Typed

// ---- Segment Entry Mode ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/segment_entry_mode.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment_value", type text}, {"segment_name", type text}, {"authorizations", type number}, {"authorized_spend", type number}, {"attempted_spend", type number}, {"decline_rate", type number}, {"challenge_rate", type number}, {"review_rate", type number}, {"alerts_per_10k", type number}, {"rule_override_rate", type number}, {"fraud_prevalence_bps", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision_at_decline", type number}, {"false_decline_rate", type number}, {"false_positive_ratio", type number}, {"fraud_loss_bps", type number}, {"mean_risk_score", type number}, {"high_risk_rate", type number}, {"cold_start_rate", type number}, {"label_coverage", type number}})
in
    Typed

// ---- Segment Tenure Band ----
let
    Source = Csv.Document(File.Contents(ExtractsFolder & "/segment_tenure_band.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"segment_value", type text}, {"segment_name", type text}, {"authorizations", type number}, {"authorized_spend", type number}, {"attempted_spend", type number}, {"decline_rate", type number}, {"challenge_rate", type number}, {"review_rate", type number}, {"alerts_per_10k", type number}, {"rule_override_rate", type number}, {"fraud_prevalence_bps", type number}, {"detection_rate", type number}, {"value_detection_rate", type number}, {"precision_at_decline", type number}, {"false_decline_rate", type number}, {"false_positive_ratio", type number}, {"fraud_loss_bps", type number}, {"mean_risk_score", type number}, {"high_risk_rate", type number}, {"cold_start_rate", type number}, {"label_coverage", type number}})
in
    Typed
