# Issue extraction schema v3 evaluation

## Scope

The evaluation used a reproducible 50-report difficult set:

- 20 reports that reached the original eight-issue ceiling;
- 10 reports missed by v2 or dominated by prospective/remedy language;
- 10 non-clinical taxonomy stress cases;
- 10 ordinary controls.

Later focused regressions covered 10 remedy/missed reports, 10 non-clinical
reports, two over-consolidated dense reports, and the single clearest dense
failure after the final Circumstances rule was changed.

## Initial 50-report result

| Stratum | Reports | v2 issues | v3 issues | v3 mean | v3 range |
|---|---:|---:|---:|---:|---:|
| Dense previous cap | 20 | 337 | 328 | 16.4 | 11–25 |
| Remedy/future-risk | 10 | 0 | 57 | 5.7 | 3–16 |
| Non-clinical | 10 | 78 | 70 | 7.0 | 4–11 |
| Ordinary control | 10 | 40 | 38 | 3.8 | 2–7 |
| **Total** | **50** | **455** | **493** | **9.86** | **2–25** |

All 50 reports succeeded and all 493 published occurrences had valid exact
evidence after deterministic recovery. No exact within-report canonical
duplicates remained. The occurrence CSV contained no report URL column; report
metadata was stored once in `01_reports.csv` and joined by `report_key`.

The net increase was driven by the intended change: every one of the 10 reports
that v2 treated as issue-free yielded supported issues. Counts in already-dense
and ordinary reports were initially slightly lower because the first v3 prompt
over-consolidated related safeguards.

## Focused findings and corrections

### Remedy-only and prospective concerns

The 20-report schema regression produced 122 issues, with 20/20 reports
represented, 122/122 valid evidence quotes, no extraction failures, and no
`other_review` values in failure state, process stage, or service sector.

Two manually known v2 misses were made permanent difficult-set regressions:

- `100065-2` produced one current system gap concerning newly qualified drivers
  carrying young passengers;
- `jacob-brown` produced one recommendation-only safeguard concerning telematics
  for young drivers.

### Non-clinical vocabulary

The first non-clinical pass used `not_stated` for 16 of 70 sector assignments.
After adding evidence-led sector values, this fell to 3 of 73. The model used
`defence_military`, `sport_leisure`, `utilities_infrastructure`,
`agriculture_animal`, `commercial_retail`, and `digital_online_services` in the
intended cases. `service_member` also replaced an incorrect custody/education
population assignment.

The remaining audit gaps led to three final contract additions:

- `animal_care_control` as a sector for domestic and companion-animal safety;
- `risk_assessment_management` as a non-clinical cross-cutting theme;
- `participant` as the population for sport and recreation participants who are
  not trainees.

### Dense-report recall

Manual comparison showed that the first v3 prompt sometimes collapsed distinct
policies, records, training duties, assessments, and notification duties under
one umbrella issue. A stronger independent-actionability split rule increased
the Gaia Pope-Sutherland report from 13 to 17 issues (v2: 18).

Some explicit systemic findings were still missed because they appeared in the
Circumstances section of a report that also had a Concerns section. The final
rule therefore admits explicitly stated findings, policy/system gaps, repeated
practices, and clear omissions from Circumstances, while continuing to exclude
chronology, outcomes, and unsupported inference. On the Alfie Gildea regression
this increased extraction from 17 to 23 supported issues (v2: 22).

## Conclusion

V3 fixes the demonstrated extraction-stage recall failure and materially improves
schema interpretability. It does not by itself establish how many recurring
types the full archive will produce: that depends on the 1,500-report extraction,
embedding/grouping thresholds, and review of the resulting recurrence groups.
The next production step is therefore a fresh complete extraction and then a
separate grouping audit, not reuse of v1/v2 occurrences.
