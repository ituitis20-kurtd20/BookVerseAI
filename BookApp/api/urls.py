from django.urls import path
from .views import semantic_search, recommend_books

urlpatterns = [
    path("semantic_search/", semantic_search, name="semantic_search"),
    path("recommend/", recommend_books, name="recommend_books"),
]