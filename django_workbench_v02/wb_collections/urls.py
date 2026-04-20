from django.urls import path

from . import views


urlpatterns = [
    path("collections/", views.collection_list, name="collection-list"),
    path("collections/<slug:collection_slug>/", views.collection_detail, name="collection-detail"),
    path("collections/<slug:collection_slug>/copy/", views.collection_copy, name="collection-copy"),
]
