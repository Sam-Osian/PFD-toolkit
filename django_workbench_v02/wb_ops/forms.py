from django import forms

from wb_workspaces.models import WorkspaceLLMProvider


class OpsPendingRunConfigForm(forms.Form):
    provider = forms.ChoiceField(
        choices=(
            (WorkspaceLLMProvider.LOCAL_OLLAMA, "Our server"),
            (WorkspaceLLMProvider.OPENAI, "OpenAI"),
        ),
        initial=WorkspaceLLMProvider.LOCAL_OLLAMA,
    )
    api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
    )
    next_url = forms.CharField(required=False)

    def clean_provider(self):
        provider = str(self.cleaned_data.get("provider") or WorkspaceLLMProvider.LOCAL_OLLAMA).strip().lower()
        if provider not in {WorkspaceLLMProvider.LOCAL_OLLAMA, WorkspaceLLMProvider.OPENAI}:
            return WorkspaceLLMProvider.LOCAL_OLLAMA
        return provider

    def clean_api_key(self):
        return str(self.cleaned_data.get("api_key") or "").strip()
