"""contractweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from contracts.views import model_form_upload
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from contracts import views



urlpatterns = [
    path('admin/', admin.site.urls),
   # path('',PDFUpload, name='upload'),
    path('',views.model_form_upload,name='home'),
    path('pdfview', views.model_form_upload,name='pdfview'),
    path('evaluation_results/<int:pk>/', views.evaluation_results,name='evaluation_results'),
    path('<int:pk>/',views.question_highlight,name="question_highlight"),
    url('output.pdf/',views.outpdf,name='outpdffile'),


]

