from django.urls import path
from . import views

# URLパターンとビュー関数を紐づける
urlpatterns = [
    path('', views.home, name='home'),  # ホーム画面（/）
    path('train/', views.train, name='train'),  # train画面（/train/）
    path('train/export/', views.export_train_zip, name='export_train_zip'),  # train.zipエクスポート
    path('val/', views.val, name='val'),  # val画面（/val/）
    path('val/export/', views.export_val_zip, name='export_val_zip'),  # val.zipエクスポート
    path('inference/', views.inference, name='inference'),  # inference画面（/inference/）
]
