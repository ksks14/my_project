from system.views import predict


def setup_routes(app=None):
    """

    :param app:
    :return:
    """
    if app:
        app.router.add_post('/predict', predict)
    assert app, 'set routes wrong! check the app'
