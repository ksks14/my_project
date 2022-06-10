from view import predict


def setup_routes(app=None):
    """
    set the app routes
    :param app:
    :return:
    """
    # by psot
    app.router.add_post('/predict', predict)
