---
title: "Capturing text spans"
description: |
  Identify the source text that supports each extracted value.
---

# Capturing text spans

Sometimes you want to know exactly which lines from the report led the model to assign a particular value to a given report's field. 

For example, say we asked the model to identify whether the deceased is a child and the model outputs `True` for a particular report, we might want to know whether this was because age is explicitly recorded (e.g. "The deceased was aged 16") or implied based on context (e.g. "The deceased was being seen by CAMHS prior to their death").

*[CAMHS]: Child and Adolescent Mental Health Services

`Extractor` can add these quotations (or 'spans') automatically. This is...

* *Great for performance*, because we're instructing the model to identify evidence for a feature value *before* it is assigned, reducing the risk of false positives.
* *Great for human verification*, because we can easily verify whether the model's evidence matches its assignment for each report.

---

## Include spans

`.extract_features()` accepts a `produce_spans` flag. When enabled, a new column starting with `spans_` is created for every feature.

For example, in our above example where we extract feature *"child"*, a separate column called *"spans_child"* will be created. Each `spans_` column contains verbatim snippets from the report which justify the extracted value.

```python
class ChildID(BaseModel):
    child: bool = Field(..., description="Whether the deceased is a child (under 18)")

result = extractor.extract_features(
    feature_model=ChildID,
    produce_spans=True,
)
result
```

The quotes returned in the spans are kept as short as possible but should always match the original text verbatim. Multiple snippets are separated with semicolons.

---

## Include & drop spans

If you're not interested in verifying the output, you might want to remove the identified spans from the returned DataFrame after extraction. Set `drop_spans=True` to remove all `spans_` columns.

As mentioned before, producing but later dropping spans is *still* likely to improve performance, because you're forcing the model to generate evidence as part of its internal workings out.


```python
extractor.extract_features(
    feature_model=DemoModel,
    produce_spans=True,
    drop_spans=True,
)
```
