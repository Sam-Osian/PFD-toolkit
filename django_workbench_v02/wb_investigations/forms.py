from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json

from django import forms
from django.db import models

from wb_notifications.models import NotificationTrigger
from wb_runs.models import RunType
from wb_workspaces.models import WorkspaceLLMProvider
from .models import InvestigationStatus


class InvestigationCreateForm(forms.Form):
    title = forms.CharField(max_length=255)
    question_text = forms.CharField(required=False, widget=forms.Textarea)
    scope_json = forms.JSONField(required=False, initial=dict)
    method_json = forms.JSONField(required=False, initial=dict)
    status = forms.ChoiceField(
        choices=InvestigationStatus.choices,
        initial=InvestigationStatus.DRAFT,
    )


class InvestigationUpdateForm(forms.Form):
    title = forms.CharField(max_length=255)
    question_text = forms.CharField(required=False, widget=forms.Textarea)
    scope_json = forms.JSONField(required=False, initial=dict)
    method_json = forms.JSONField(required=False, initial=dict)
    status = forms.ChoiceField(choices=InvestigationStatus.choices)


class TemporalScopeOption(models.TextChoices):
    ALL_REPORTS = "all_reports", "All reports"
    LAST_3_YEARS = "last_3_years", "Last 3 years"
    LAST_YEAR = "last_year", "Last year"
    LAST_6_MONTHS = "last_6_months", "Last 6 months"
    MOST_RECENT_100 = "most_recent_100", "100 most recent reports"
    CUSTOM_RANGE = "custom_range", "Custom date range"


WIZARD_STAGES = (
    "question",
    "scope",
    "method",
    "filter",
    "themes",
    "extract",
    "review",
)


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        # Handles leap-day rollover.
        return value.replace(month=2, day=28, year=value.year - years)


def temporal_scope_parameters(
    *,
    scope_option: str,
    today: date | None = None,
    custom_start_date: date | None = None,
    custom_end_date: date | None = None,
) -> dict:
    current_date = today or date.today()
    option = str(scope_option or "").strip()

    if option == TemporalScopeOption.LAST_3_YEARS:
        return {
            "query_start_date": _subtract_years(current_date, 3),
            "query_end_date": current_date,
            "report_limit": None,
            "scope_option": option,
        }
    if option == TemporalScopeOption.LAST_YEAR:
        return {
            "query_start_date": _subtract_years(current_date, 1),
            "query_end_date": current_date,
            "report_limit": None,
            "scope_option": option,
        }
    if option == TemporalScopeOption.LAST_6_MONTHS:
        return {
            "query_start_date": current_date - timedelta(days=183),
            "query_end_date": current_date,
            "report_limit": None,
            "scope_option": option,
        }
    if option == TemporalScopeOption.MOST_RECENT_100:
        return {
            "query_start_date": None,
            "query_end_date": None,
            "report_limit": 100,
            "scope_option": option,
        }
    if option == TemporalScopeOption.CUSTOM_RANGE and custom_start_date and custom_end_date:
        return {
            "query_start_date": custom_start_date,
            "query_end_date": custom_end_date,
            "report_limit": None,
            "scope_option": option,
        }
    return {
        "query_start_date": None,
        "query_end_date": None,
        "report_limit": None,
        "scope_option": TemporalScopeOption.ALL_REPORTS,
    }


def build_pipeline_plan(
    *,
    run_filter: bool,
    run_themes: bool,
    run_extract: bool,
) -> list[str]:
    plan: list[str] = []
    if run_filter:
        plan.append(RunType.FILTER)
    if run_themes:
        plan.append(RunType.THEMES)
    if run_extract:
        plan.append(RunType.EXTRACT)
    return plan


class InvestigationWizardQuestionForm(forms.Form):
    title = forms.CharField(max_length=255)
    question_text = forms.CharField(required=False, widget=forms.Textarea)


class InvestigationWizardScopeForm(forms.Form):
    scope_option = forms.ChoiceField(
        choices=TemporalScopeOption.choices,
        initial=TemporalScopeOption.ALL_REPORTS,
    )
    custom_start_date = forms.DateField(required=False)
    custom_end_date = forms.DateField(required=False)

    def clean(self):
        cleaned = super().clean()
        option = str(cleaned.get("scope_option") or "").strip()
        start_date = cleaned.get("custom_start_date")
        end_date = cleaned.get("custom_end_date")
        if option == TemporalScopeOption.CUSTOM_RANGE:
            if not start_date or not end_date:
                raise forms.ValidationError("Custom date range requires a start date and end date.")
            if end_date < start_date:
                raise forms.ValidationError("End date must be on or after start date.")
        return cleaned

    def resolved_scope(self, *, today: date | None = None) -> dict:
        option = str(self.cleaned_data.get("scope_option") or "")
        return temporal_scope_parameters(
            scope_option=option,
            today=today,
            custom_start_date=self.cleaned_data.get("custom_start_date"),
            custom_end_date=self.cleaned_data.get("custom_end_date"),
        )


class InvestigationWizardMethodForm(forms.Form):
    run_filter = forms.BooleanField(required=False, initial=True)
    run_themes = forms.BooleanField(required=False, initial=False)
    run_extract = forms.BooleanField(required=False, initial=False)

    def clean(self):
        cleaned = super().clean()
        if not any(
            [
                bool(cleaned.get("run_filter")),
                bool(cleaned.get("run_themes")),
                bool(cleaned.get("run_extract")),
            ]
        ):
            raise forms.ValidationError("Select at least one stage to run.")
        return cleaned

    def pipeline_plan(self) -> list[str]:
        return build_pipeline_plan(
            run_filter=bool(self.cleaned_data.get("run_filter")),
            run_themes=bool(self.cleaned_data.get("run_themes")),
            run_extract=bool(self.cleaned_data.get("run_extract")),
        )


class InvestigationWizardThemesConfigForm(forms.Form):
    enabled = forms.BooleanField(required=False, initial=False)
    seed_topics = forms.CharField(required=False, widget=forms.Textarea)
    min_themes = forms.IntegerField(required=False, min_value=1, max_value=100)
    max_themes = forms.IntegerField(required=False, min_value=1, max_value=100)
    extra_theme_instructions = forms.CharField(required=False, widget=forms.Textarea)

    def clean(self):
        cleaned = super().clean()
        if not bool(cleaned.get("enabled")):
            return cleaned
        min_themes = cleaned.get("min_themes")
        max_themes = cleaned.get("max_themes")
        if min_themes is not None and max_themes is not None and min_themes > max_themes:
            raise forms.ValidationError("Minimum themes cannot exceed maximum themes.")
        return cleaned


class InvestigationWizardExtractConfigForm(forms.Form):
    enabled = forms.BooleanField(required=False, initial=False)
    feature_fields = forms.CharField(
        required=False,
        widget=forms.Textarea,
        label="Feature fields (name | description | type)",
        help_text=(
            "One per line, e.g. age_at_death | Age at death in years | integer. "
            "JSON list is also accepted for compatibility."
        ),
    )
    allow_multiple = forms.BooleanField(required=False, initial=False)
    force_assign = forms.BooleanField(required=False, initial=False)
    skip_if_present = forms.BooleanField(required=False, initial=True)
    extract_include_supporting_quotes = forms.BooleanField(required=False, initial=False)

    def _parse_feature_fields(self, raw_value) -> list[dict]:
        if isinstance(raw_value, list):
            return raw_value
        text = str(raw_value or "").strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise forms.ValidationError("Invalid JSON for feature fields.") from exc
            if not isinstance(parsed, list):
                raise forms.ValidationError("Feature fields JSON must be a list.")
            return parsed

        rows: list[dict] = []
        for line in text.splitlines():
            item = line.strip()
            if not item:
                continue
            if "|" in item:
                parts = [part.strip() for part in item.split("|")]
                if len(parts) < 3:
                    raise forms.ValidationError(
                        "Each feature line must include name | description | type."
                    )
                name = parts[0]
                description = parts[1]
                field_type = parts[2]
            elif ":" in item:
                # Legacy shorthand: name:type
                name, field_type = item.split(":", 1)
                description = ""
            elif "," in item:
                # Legacy shorthand: name,type
                name, field_type = item.split(",", 1)
                description = ""
            else:
                raise forms.ValidationError(
                    "Each feature line must be name | description | type."
                )
            rows.append(
                {
                    "name": name.strip(),
                    "description": description.strip(),
                    "type": field_type.strip(),
                }
            )
        return rows

    def clean(self):
        cleaned = super().clean()
        if not bool(cleaned.get("enabled")):
            return cleaned

        feature_fields = self._parse_feature_fields(cleaned.get("feature_fields"))
        if not isinstance(feature_fields, list) or not feature_fields:
            raise forms.ValidationError(
                "Extract configuration requires at least one feature field."
            )
        for index, row in enumerate(feature_fields, start=1):
            if not isinstance(row, dict):
                raise forms.ValidationError(f"Feature row {index} must be an object.")
            name = str(row.get("name") or row.get("field_name") or "").strip()
            description = str(row.get("description") or "").strip()
            field_type = str(row.get("type") or "").strip()
            if not name:
                raise forms.ValidationError(f"Feature row {index} is missing a field name.")
            if not description:
                raise forms.ValidationError(f"Feature row {index} is missing a description.")
            if not field_type:
                raise forms.ValidationError(f"Feature row {index} is missing a type.")
        cleaned["feature_fields"] = feature_fields
        return cleaned


class InvestigationWizardReviewForm(forms.Form):
    execution_mode = forms.ChoiceField(
        choices=(("real", "Real"),),
        initial="real",
        required=True,
    )
    provider = forms.ChoiceField(
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
        required=False,
    )
    model_name = forms.CharField(required=False, initial="gpt-4.1-mini")
    max_parallel_workers = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=32,
        initial=1,
    )
    request_completion_email = forms.BooleanField(required=False, initial=True)
    notify_on = forms.ChoiceField(
        required=False,
        choices=NotificationTrigger.choices,
        initial=NotificationTrigger.ANY,
    )
    api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=True, attrs={"autocomplete": "off"}),
        label="API key",
    )
    base_url = forms.URLField(required=False, label="Provider base URL override")
    save_api_key = forms.BooleanField(required=False, initial=True)


@dataclass
class InvestigationWizardState:
    stage: str = "question"
    title: str = ""
    question_text: str = ""
    scope_option: str = TemporalScopeOption.ALL_REPORTS
    scope_start_date: str = ""
    scope_end_date: str = ""
    run_filter: bool = True
    run_themes: bool = False
    run_extract: bool = False
    filter_config: dict = field(default_factory=dict)
    themes_config: dict = field(default_factory=dict)
    extract_config: dict = field(default_factory=dict)
    review_config: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict | None) -> "InvestigationWizardState":
        raw = payload if isinstance(payload, dict) else {}
        stage = str(raw.get("stage") or "question").strip().lower()
        if stage not in WIZARD_STAGES:
            stage = "question"
        return cls(
            stage=stage,
            title=str(raw.get("title") or "").strip(),
            question_text=str(raw.get("question_text") or "").strip(),
            scope_option=str(raw.get("scope_option") or TemporalScopeOption.ALL_REPORTS).strip(),
            scope_start_date=str(raw.get("scope_start_date") or "").strip(),
            scope_end_date=str(raw.get("scope_end_date") or "").strip(),
            run_filter=bool(raw.get("run_filter", True)),
            run_themes=bool(raw.get("run_themes")),
            run_extract=bool(raw.get("run_extract")),
            filter_config=raw.get("filter_config") if isinstance(raw.get("filter_config"), dict) else {},
            themes_config=raw.get("themes_config") if isinstance(raw.get("themes_config"), dict) else {},
            extract_config=raw.get("extract_config")
            if isinstance(raw.get("extract_config"), dict)
            else {},
            review_config=raw.get("review_config") if isinstance(raw.get("review_config"), dict) else {},
        )

    def to_json(self) -> dict:
        return {
            "stage": self.stage,
            "title": self.title,
            "question_text": self.question_text,
            "scope_option": self.scope_option,
            "scope_start_date": self.scope_start_date,
            "scope_end_date": self.scope_end_date,
            "run_filter": self.run_filter,
            "run_themes": self.run_themes,
            "run_extract": self.run_extract,
            "filter_config": self.filter_config,
            "themes_config": self.themes_config,
            "extract_config": self.extract_config,
            "review_config": self.review_config,
        }

    def pipeline_plan(self) -> list[str]:
        return build_pipeline_plan(
            run_filter=bool(self.run_filter),
            run_themes=bool(self.run_themes),
            run_extract=bool(self.run_extract),
        )


class InvestigationWizardFilterConfigForm(forms.Form):
    enabled = forms.BooleanField(required=False, initial=True)
    search_query = forms.CharField(required=False, widget=forms.Textarea)
    filter_df = forms.BooleanField(required=False, initial=True)
    include_supporting_quotes = forms.BooleanField(required=False, initial=False)
    coroner_filters = forms.CharField(required=False)
    area_filters = forms.CharField(required=False)
    receiver_filters = forms.CharField(required=False)

    @staticmethod
    def _parse_csv_values(raw_value: str) -> list[str]:
        return [
            value.strip()
            for value in str(raw_value or "").split(",")
            if str(value or "").strip()
        ]

    def clean(self):
        cleaned = super().clean()
        if not bool(cleaned.get("enabled")):
            return cleaned
        query = str(cleaned.get("search_query") or "").strip()
        if not query:
            raise forms.ValidationError("Filter step requires a search query.")
        cleaned["search_query"] = query
        cleaned["selected_filters"] = {
            "coroner": self._parse_csv_values(cleaned.get("coroner_filters") or ""),
            "area": self._parse_csv_values(cleaned.get("area_filters") or ""),
            "receiver": self._parse_csv_values(cleaned.get("receiver_filters") or ""),
        }
        return cleaned


class InvestigationExportForm(forms.Form):
    bundle_name = forms.CharField(required=False, max_length=255)
    download_include_dataset = forms.BooleanField(required=False, initial=True)
    download_include_excluded = forms.BooleanField(required=False, initial=True)
    download_include_theme = forms.BooleanField(required=False, initial=True)
    download_include_feature_grid = forms.BooleanField(required=False, initial=True)
    download_include_script = forms.BooleanField(required=False, initial=False)
    latest_per_artifact_type = forms.BooleanField(required=False, initial=True)
    max_artifacts = forms.IntegerField(required=False, min_value=1, max_value=500)
    request_completion_email = forms.BooleanField(required=False, initial=False)
    notify_on = forms.ChoiceField(
        required=False,
        choices=NotificationTrigger.choices,
        initial=NotificationTrigger.ANY,
    )

    def clean(self):
        cleaned = super().clean()
        if not any(
            [
                bool(cleaned.get("download_include_dataset")),
                bool(cleaned.get("download_include_excluded")),
                bool(cleaned.get("download_include_theme")),
                bool(cleaned.get("download_include_feature_grid")),
                bool(cleaned.get("download_include_script")),
            ]
        ):
            raise forms.ValidationError("Select at least one export component.")
        return cleaned
