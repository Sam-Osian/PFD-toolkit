from django import forms

from wb_workspaces.models import WorkspaceLLMProvider


class OpsPendingRunConfigForm(forms.Form):
    provider = forms.ChoiceField(
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
    )
    api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
    )
    next_url = forms.CharField(required=False)

    def clean_provider(self):
        provider = str(self.cleaned_data.get("provider") or WorkspaceLLMProvider.OPENAI).strip().lower()
        if provider not in {WorkspaceLLMProvider.OPENAI, WorkspaceLLMProvider.OPENROUTER}:
            return WorkspaceLLMProvider.OPENAI
        return provider

    def clean_api_key(self):
        return str(self.cleaned_data.get("api_key") or "").strip()
