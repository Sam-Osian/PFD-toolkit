from django import forms

from wb_notifications.models import NotificationTrigger
from wb_workspaces.models import WorkspaceLLMProvider

from .models import RunType


class RunQueueForm(forms.Form):
    run_type = forms.ChoiceField(choices=RunType.choices)
    provider = forms.ChoiceField(
        required=False,
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
        help_text="Provider used for real workflow runs.",
    )
    model_name = forms.CharField(
        required=False,
        initial="gpt-4.1-mini",
        help_text="LLM model name used for this run.",
    )
    api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
        help_text="Leave blank to use your saved workspace key.",
    )
    base_url = forms.URLField(
        required=False,
        help_text="Optional custom API base URL override for this provider.",
    )
    save_api_key = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Save key for this workbook and provider.",
    )
    input_config_json = forms.JSONField(required=False, initial=dict)
    query_start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    query_end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    request_completion_email = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Email me when this run reaches a terminal status.",
    )
    notify_on = forms.ChoiceField(
        required=False,
        choices=NotificationTrigger.choices,
        initial=NotificationTrigger.ANY,
        help_text="When to send email if completion notification is enabled.",
    )

    def clean(self):
        cleaned = super().clean()
        config = cleaned.get("input_config_json")
        if not isinstance(config, dict):
            config = {}
        provider = (cleaned.get("provider") or WorkspaceLLMProvider.OPENAI).strip().lower()
        if provider not in {WorkspaceLLMProvider.OPENAI, WorkspaceLLMProvider.OPENROUTER}:
            provider = WorkspaceLLMProvider.OPENAI

        model_name = (cleaned.get("model_name") or "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        base_url = (cleaned.get("base_url") or "").strip()
        config["provider"] = provider
        config["model_name"] = model_name
        if base_url:
            config_key = "openrouter_base_url" if provider == WorkspaceLLMProvider.OPENROUTER else "openai_base_url"
            config[config_key] = base_url

        cleaned["input_config_json"] = config
        cleaned["provider"] = provider
        cleaned["model_name"] = model_name
        cleaned["base_url"] = base_url
        return cleaned


class RunCancelForm(forms.Form):
    cancel_reason = forms.CharField(required=False, widget=forms.Textarea)
