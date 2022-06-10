from os import system

from django.shortcuts import render, redirect
from django.urls import reverse
from system.models import User
from threading import Thread


# Create your views here.
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        user = User.objects.filter(username=username).first()
        if User:
            password = request.POST['password']
            if user.password == password:
                # 设置session
                request.session['user_id'] = user.id
                request.session.set_expiry(60 * 60)
                return redirect(reverse('index'))
            else:
                return redirect(reverse('index'))
    return render(request, 'login.html')


def index(request):
    """

    :param request:
    :return:
    """
    return render(request, 'index.html')


def GUI_video(request):
    """
    从这里的情况来看，应该是要实现一个异步类的重写
    :param :
    :return:
    """
    order_1 = 'python ./my_GUI/my_tkinter.py '
    t_2 = Thread(target=system, args=(order_1, ))
    t_2.start()
    return render(request, 'index.html')



def more(request):
    """

    :param request:
    :return:
    """
    return render(request, 'basic_gallery.html')


def profile(request):
    """

    :param request:
    :return:
    """
    return render(request, 'login.html')