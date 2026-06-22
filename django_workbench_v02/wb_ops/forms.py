from django import forms

from wb_workspaces.models import WorkspaceLLMProvider


OPS_OPENAI_MODEL_CHOICES = (
    ("gpt-4.1-mini", "gpt-4.1-mini"),
    ("gpt-4.1", "gpt-4.1"),
)


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
    model_name = forms.ChoiceField(
        choices=OPS_OPENAI_MODEL_CHOICES,
        required=False,
        initial="gpt-4.1-mini",
    )
    next_url = forms.CharField(required=False)

    def clean_provider(self):
        provider = str(self.cleaned_data.get("provider") or WorkspaceLLMProvider.LOCAL_OLLAMA).strip().lower()
        if provider not in {WorkspaceLLMProvider.LOCAL_OLLAMA, WorkspaceLLMProvider.OPENAI}:
            return WorkspaceLLMProvider.LOCAL_OLLAMA
        return provider

    def clean_api_key(self):
        return str(self.cleaned_data.get("api_key") or "").strip()

    def clean_model_name(self):
        provider = str(self.cleaned_data.get("provider") or WorkspaceLLMProvider.LOCAL_OLLAMA).strip().lower()
        model_name = str(self.cleaned_data.get("model_name") or "").strip()
        if provider != WorkspaceLLMProvider.OPENAI:
            return ""
        if model_name not in dict(OPS_OPENAI_MODEL_CHOICES):
            return "gpt-4.1-mini"
        return model_name
