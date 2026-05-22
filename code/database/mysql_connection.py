from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def create_mysql_server_engine(
    user: str,
    password: str,
    host: str,
    port: str,
) -> Engine:
    """
    创建 MySQL Server 连接，不指定具体 database。

    Args:
        user: MySQL 用户名。
        password: MySQL 密码。
        host: MySQL 主机地址。
        port: MySQL 端口号。

    Returns:
        MySQL Server 连接引擎。
    """
    return create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}"
    )


def create_database(
    server_engine: Engine,
    database_name: str,
) -> None:
    """
    如果数据库不存在，则创建数据库。

    Args:
        server_engine: MySQL Server 连接引擎。
        database_name: 数据库名称。
    """
    with server_engine.connect() as conn:
        conn.execute(
            text(
                f"""
                CREATE DATABASE IF NOT EXISTS {database_name}
                DEFAULT CHARACTER SET utf8mb4
                """
            )
        )
        conn.commit()


def create_database_engine(
    user: str,
    password: str,
    host: str,
    port: str,
    database_name: str,
) -> Engine:
    """
    创建指定 database 的 MySQL 连接。

    Args:
        user: MySQL 用户名。
        password: MySQL 密码。
        host: MySQL 主机地址。
        port: MySQL 端口号。
        database_name: 数据库名称。

    Returns:
        指定数据库的连接引擎。
    """
    return create_engine(
        f"mysql+pymysql://{user}:{password}"
        f"@{host}:{port}/{database_name}"
        f"?charset=utf8mb4"
    )