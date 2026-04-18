from django import forms

from .models import Workspace


class WorkspaceCreateForm(forms.ModelForm):
    class Meta:
        model = Workspace
        fields = ["title", "slug", "description", "visibility", "is_listed"]
