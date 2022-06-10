from sqlalchemy import create_engine, MetaData
from system.settings import config
from system.db import user

sql_DSN = 'postgresql://{user}:{password}@{host}:{port}/{database}'


def create_tables(engine=create_engine(sql_DSN.format(**config['postgres']))):
    """
    create the tables
    :param engine:
    :return:
    """
    meta = MetaData()
    meta.create_all(bind=engine, tables=[user, ])

def drop_tables(engine=create_engine(sql_DSN.format(**config['postgres']))):
    """

    :return:
    """
    meta = MetaData()
    meta.drop_all(bind=engine, tables=[user, ])


if __name__ == '__main__':
    # create tables
    # create_tables()
    # drop_tables()
    pass
