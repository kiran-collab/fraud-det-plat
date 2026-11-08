# Spotfire custom expressions

Every governed metric, as the Spotfire expression that reproduces it from
`Decision Detail`. **Use these for drill-down only.** The headline figure
should always come from the `KPI Overall` table, which carries the value the
Python layer computed under test. These expressions exist so an analyst can
break a number down by a dimension the extracts do not pre-aggregate - not so
the number can be re-derived.

| Metric | Owner | Spotfire expression over `Decision Detail` |
| --- | --- | --- |
| Decline precision | Data Science | `Sum(If([is_caught_fraud],1,0)) / Sum(If([is_declined],1,0))` |
| Detection rate (recall) | Data Science | `Sum(If([is_caught_fraud],1,0)) / Sum(If([is_fraud],1,0))` |
| False decline rate | Fraud Strategy | `Sum(If([is_false_decline],1,0)) / Sum(If([is_fraud],0,1))` |
| Fraud prevalence | Fraud Strategy | `10000 * Sum(If([is_fraud],1,0)) / Count()` |
| Value detection rate | Data Science | `Sum([fraud_value_stopped]) / (Sum([fraud_value_stopped])+Sum([fraud_value_lost]))` |
| Decline rate | Fraud Strategy | `Sum(If([is_declined],1,0)) / Count()` |
| Review rate | Fraud Operations | `Sum(If([action]='review',1,0)) / Count()` |
| Rule escalation rate | Fraud Strategy | `Sum(If([triggered_rule_count]>0,1,0)) / Count()` |
| Step-up rate | Fraud Strategy | `Sum(If([action]='step_up',1,0)) / Count()` |
| Alerts per 10k authorizations | Fraud Operations | `10000 * Sum(If([action]='review',1,0)) / Count()` |
| False positives per catch | Fraud Operations | `Sum(If([is_false_decline],1,0)) / Sum(If([is_caught_fraud],1,0))` |
| Fraud loss (bps of spend) | Fraud Strategy | `10000 * Sum([fraud_value_lost]) / Sum([amount])` |
| Cold-start rate | Data Science | `Sum(If([sequence_score]=0.5,1,0)) / Count()` |
| High-risk rate | Data Science | `Sum(If([risk_score]>0.55,1,0)) / Count()` |
| Label coverage | Data Governance | `Sum(If([is_fraud] is null,0,1)) / Count()` |
| Mean risk score | Data Science | `Avg([risk_score])` |
| Attempted spend | Fraud Strategy | `Sum([amount])` |
| Authorizations | Fraud Strategy | `Count()` |
| Authorized spend | Fraud Strategy | `Sum(If([action]<>'decline',[amount],0))` |

## Why these are not the source of truth

`value_detection_rate` above divides stopped value by total fraud value. The
Python definition does the same thing - but it also decides what counts as
settled, which rows are in the holdout, and how a null label is treated. Those
decisions are tested; a cell in a Spotfire expression editor is not. Where the
two disagree, the disagreement is the bug, and the fix belongs in
`src/fraudplat/analytics/metrics.py`.
