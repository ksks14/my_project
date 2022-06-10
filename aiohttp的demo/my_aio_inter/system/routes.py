from views import index, login, register, delete_user, show_all, pre_img



def setup_routes(app):
    """
    set the routes
    :param app:
    :return:
    """
    app.router.add_get('/', index)
    app.router.add_get('/show_all', show_all, name='show_all')
    app.router.add_get('/register', register, name='register')
    app.router.add_get('/login', login, name='login')
    app.router.add_get('/system/delete', delete_user, name='delete_user')
    app.router.add_post('/system/pre_img', pre_img, name='pre_img')