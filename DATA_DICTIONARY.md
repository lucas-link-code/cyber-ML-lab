# Synthetic CSV data dictionary

All CSVs are UTF-8 with headers. Fixed generator base seed: `20261031`; dataset-specific seeded NumPy generators. Unless specified otherwise, counts are nonnegative; `1` indicates the positive flag. `NaN`/blank values are intentional missing telemetry. **Never use IDs, labels, or future/post-investigation fields as ordinary ML features.**

## `phishing_emails.csv` (6,000 rows)

| Column | Meaning / range | Modelling role |
| --- | --- | --- |
| `email_id` | Unique synthetic `MAIL-######` email record key | Identifier; exclude |
| `sender_domain` | Synthetic `.example` source domain | Categorical input; not a real URL/IOC |
| `sender_reputation` | Simulated 1–100 reputation score; can be blank | Numeric feature |
| `spf_result` | `pass`, `fail`, `softfail` or `none` authentication result; sometimes blank | Categorical feature |
| `dkim_result` | `pass`, `fail` or `none` | Categorical feature |
| `dmarc_result` | `pass`, `fail` or `none`; sometimes blank | Categorical feature |
| `url_count` | Total number of links in email | Numeric feature |
| `suspicious_url_count` | Number of links flagged by synthetic pre-delivery URL assessment | Numeric feature; may be wrong |
| `attachment_count` | Total attachments | Numeric feature |
| `has_shortened_url` | `1` if a shortener-type link was indicated | Numeric/binary feature |
| `urgent_language_score` | Synthetic 0–1 urgency indicator | Numeric feature |
| `body_length` | Simulated message length in characters; sometimes blank | Numeric feature |
| `subject_length` | Simulated subject length | Numeric feature |
| `subject_text` | Safe, generated sample subject | TF-IDF text feature |
| `body_text` | Safe, generated message body | TF-IDF text feature |
| `message_theme` | Random colour theme, uncorrelated with phishing | Intentional irrelevant feature; excluded |
| `phishing_label` | `1` synthetic phishing, `0` legitimate | **Target; exclude from inputs** |

`phishing_emails_leakage_challenge.csv` contains all columns above plus `post_investigation_decision`, a nearly label-identical outcome known only **after** investigation. This column is intentionally invalid for pre-delivery detection and is excluded from regular model feature lists. `leakage_demo.py` uses it **solely to demonstrate leakage**.

## `edr_events.csv` (6,500 rows)

| Column | Meaning / range | Modelling role |
| --- | --- | --- |
| `event_id` | Unique `EDR-######` identifier | Exclude |
| `hostname` | Synthetic endpoint ID with repeated events | Group splitting only; exclude from model |
| `endpoint_type` | `user`, `developer`, `server` | Categorical feature and population grouping |
| `process_name` | Generated common/LOLBIN-like Windows process basename | Categorical feature |
| `parent_process` | Generated parent image basename; can be blank | Categorical feature |
| `command_line_length` | Character count of simulated command line | Numeric feature |
| `process_prevalence` | Synthetic proportion 0–1 of similar process prevalence; can be blank | Numeric feature |
| `network_connection_count` | Connections initiated in observation window | Numeric feature |
| `unique_destination_count` | Distinct destinations observed | Numeric feature |
| `outbound_bytes` | Simulated bytes sent by process; can be blank | Numeric feature, log-transformed in scripts |
| `process_execution_frequency` | Execution count in observation window | Numeric feature |
| `unsigned_binary` | `1` if unsigned | Binary feature |
| `unusual_parent_child` | `1` for synthetic unusual process ancestry | Binary feature |
| `random_trace_tag` | Arbitrary event number with no predictive meaning | Irrelevant/noise feature (used only in noisy unsupervised feature set) |
| `malicious_label` | `1` malicious, `0` benign | **Classification target / anomaly evaluation-only label** |

`edr_events_challenge.csv` has the same schema with a different random cohort, including more benign developer processes with high network activity, long command lines and unsigned binaries.

## `alert_triage.csv` (3,500 rows)

| Column | Meaning / range | Modelling role |
| --- | --- | --- |
| `alert_id` | Unique `ALERT-######` record identifier | Exclude |
| `alert_type` | `phishing`, `malware`, `identity`, `edr`, `network` or `data_loss` | Categorical feature |
| `alert_severity` | `low`, `medium`, `high` or `critical` | Categorical feature |
| `telemetry_count` | Relevant telemetry records collected | Numeric feature |
| `related_host_count` | Number of related hosts | Numeric feature |
| `historical_alert_frequency` | Approximate historical recurrence count; can be blank | Numeric feature |
| `analyst_workload` | Number of competing alerts | Numeric feature |
| `investigation_complexity` | Synthetic 0–10 difficulty score; can be blank | Numeric feature |
| `collection_day` | Random integer day-of-month for distracting/uninformative context | Irrelevant feature; excluded |
| `estimated_triage_minutes` | Simulated true elapsed triage time in minutes | **Regression target; exclude from inputs** |

`alert_triage_challenge.csv` has additional measurement noise and extra time associated with data-loss events. It is a separate cohort rather than a held-out portion of the base CSV.

## `network_activity.csv` (5,000 rows)

| Column | Meaning / range | Modelling role |
| --- | --- | --- |
| `session_id` | Unique `NET-######` record identifier | Exclude |
| `source_host` | Synthetic host name with repeated sessions | Group splitting / reporting only; exclude |
| `host_role` | `user`, `developer` or `server` | Population group for separate anomaly baselines; exclude from distance vectors |
| `connection_frequency` | Number of connections per observation window | Numeric feature |
| `destination_count` | Unique destinations reached | Numeric feature |
| `bytes_sent` | Synthetic bytes transmitted | Numeric feature, log-transformed |
| `bytes_received` | Synthetic bytes received; can be blank | Numeric feature, log-transformed |
| `destination_port` | TCP-like service port number | Numeric feature in raw set; an imperfect distance encoding |
| `connection_duration` | Connection duration in seconds; can be blank | Numeric feature |
| `unusual_hour_activity` | `1` if outside usual hours | Binary feature |
| `failed_connection_count` | Connection failure count | Numeric feature |
| `random_trace_tag` | Non-predictive random ID-like integer | Irrelevant feature; noisy set only |
| `known_anomaly_label` | `1` synthetic ground-truth anomaly, `0` routine | **Evaluation-only; NEVER used in unsupervised fitting, preprocessing or threshold selection** |

`network_activity_challenge.csv` includes more legitimate developer sessions with higher-volume network activity, increasing the potential for population-related false positives.

## Experimental data caveats

- Target correlations and hard cases are engineered, not learned from real enterprise telemetry. Synthetic labels are generated **before** observables to create overlapping class-conditional distributions; labels are withheld from unsupervised training.
- Some synthetic subjects repeat and the body is drawn from safe phrase templates. They are suitable for teaching TF-IDF and feature pipelines but not for measuring real-world NLP capability.
- `spf_result`, `dkim_result` and `dmarc_result` represent simulated authentication outcomes rather than actual protocol validation; `suspicious_url_count` represents a pre-delivery synthetic enrichment signal, not a downstream analyst verdict.
- The regression target is the generated elapsed triage time; the name `estimated_triage_minutes` is part of the requested dataset schema and does not signify a known inference-time input.
- Integer `destination_port` in Euclidean distance is pedagogically imperfect; dropping it via `--features behavioral` demonstrates feature selection and distance limitations. Numerical outputs remain synthetic.
