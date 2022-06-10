from sqlalchemy import MetaData, Table, Column, ForeignKey, Integer, String, Date
from aiopg.sa import create_engine

__all__ = ['user', ]

# 实例化metadata对象， 该对象包含了一系列表对象，属于字典对象
meta = MetaData()

# 创建用户表
user = Table(
    'user', meta,
    Column('id', Integer, primary_key=True),
    Column('username', String(200), nullable=False),
    Column('password', String(200), nullable=False),
    Column('name', String(200), nullable=False),
)


async def pg_context(app):
    """

    :param app:
    :return:
    """
    conf = app['conf']['postgres']
    engine = await create_engine(
        database=conf['database'],
        user=conf['user'],
        password=conf['password'],
        host=conf['host'],
        port=conf['port'],
        minsize=conf['minsize'],
        maxsize=conf['maxsize'],
    )
    app['db'] = engine
    yield
    app['db'].close()
    await app['db'].wait_closed()


async def show_all_data(conn, data):
    """

    :param conn:
    :param data:
    :return:
    """
    async for row in conn.execute(user.select()):
        print(row.id, row.username)


async def register_in(conn, data):
    """
    write the data into the database
    :param conn:
    :param data:
    :return:
    """
    result = await conn.execute(
        user.insert().values(username=data['username'], password=data['password'], name=data['name'])
    )
    async for row in conn.execute(user.select()):
        print(row.id, row.username)


async def login_check(conn, data):
    """
    一次循环直接return
    :param conn:
    :param data:
    :return:
    """
    try:
        async for row in conn.execute(
                user.select().where(user.c.username == data['username'] and user.c.password == data['password'])):
            return row
    except Exception as e:
        return None


async def delete_data(conn, data):
    """
    delete the table ele
    :param conn:
    :param data:
    :return:
    """
    try:
        await conn.execute(
            user.delete().where(
                user.c.id == data['id'] and user.c.username == data['username'] and user.c.password == data['password'])
        )
        return True
    except Exception as e:
        return None
