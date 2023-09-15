from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('',views.SignupPage,name='signup'),
    path('login/',views.LoginPage,name='login'),
    path('home/',views.HomePage,name='home'),
    path('logout/',views.LogoutPage,name='logout'),
    path('select/',views.select_image,name='select_image'),
    path('process-image/',views.process_image,name='process_image'),
    path('download-image/', views.download_image, name='download-image'),


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)