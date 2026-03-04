# Receiver Themes

Temporary utility for generating rule-based receiver collections as `theme_*` boolean columns.

Themes included:

- `theme_sent_to_nhs_bodies`
  Matches receiver segments containing `NHS`, `Integrated Care Board`, or `Health Board`.

- `theme_sent_to_government_departments`
  Matches segments starting with `Department of` or `Department for`, plus named departments such as `Cabinet Office`, `Home Office`, `Ministry of Justice`, `Attorney General's Office`, and `Welsh Government`.

- `theme_sent_to_prisons`
  Matches prison-specific segments such as `HMP`, `HM Prison`, `Prison`, `Young Offender Institution`, `YOI`, `Secure Training Centre`, and `HM Prison and Probation Service`.

- `theme_sent_to_health_regulators`
  Matches expanded regulator names including `Care Quality Commission`, `National Institute for Health and Care Excellence`, `Medicines and Healthcare Products Regulatory Agency`, `General Medical Council`, `Nursing and Midwifery Council`, `Health and Care Professions Council`, and `General Pharmaceutical Council`.

- `theme_sent_to_local_government`
  Matches local-government bodies such as `County Council`, `City Council`, `Borough Council`, `District Council`, `County Borough Council`, `Metropolitan Borough Council`, `London Borough of ...`, `Unitary Authority`, `Local Authority`, and council names that end in `Council` unless they match known non-local authority exclusions such as `General Medical Council` or `National Police Chiefs Council`.

Notes:

- Matching is applied per semicolon-separated receiver segment.
- Rules run on normalized receiver text only; they do not use the report body.
- Collections are intentionally broad enough for browse dashboards, not individual-recipient filtering.
