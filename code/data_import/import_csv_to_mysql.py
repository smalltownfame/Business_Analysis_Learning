import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

# 将 code/ 目录加入 Python 模块搜索路径，便于导入自定义模块
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from database.mysql_connection import (
    create_database,
    create_database_engine,
    create_mysql_server_engine,
)


# =========================
# 0. 读取环境变量
# =========================

load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
DATABASE_NAME = os.getenv("DATABASE_NAME")

required_env_vars = {
    "MYSQL_USER": MYSQL_USER,
    "MYSQL_PASSWORD": MYSQL_PASSWORD,
    "MYSQL_HOST": MYSQL_HOST,
    "MYSQL_PORT": MYSQL_PORT,
    "DATABASE_NAME": DATABASE_NAME,
}

missing_env_vars = [
    key for key, value in required_env_vars.items() if value is None
]
if missing_env_vars:
    raise ValueError(f"缺少环境变量：{missing_env_vars}")


# =========================
# 1. 读取 CSV
# =========================

csv_path = "data/raw/data_min.csv"
df = pd.read_csv(csv_path)


print("数据前5行：")
print(df.head())

print("\n字段信息：")
print(df.info())


# =========================
# 2. 数据质量检查
# =========================

expected_columns = [
    "user_id",
    "item_id",
    "behavior_type",
    "item_category",
    "time",
]

missing_columns = [
    col for col in expected_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"缺少字段：{missing_columns}")

print("\n缺失值统计：")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"\n完全重复行数量：{duplicate_count}")

print("\nbehavior_type 分布：")
print(df["behavior_type"].value_counts(dropna=False))


# =========================
# 3. 数据清洗与字段转换
# =========================

# 用户行为字段‘behavior_type’ 应只包含：
# 1 = 浏览(pv), 2 = 收藏(fav), 3 = 加购物车(cart), 4 = 购买(buy)
df["behavior_type"] = pd.to_numeric(
    df["behavior_type"],
    errors="coerce",
)

invalid_behavior = df[
    ~df["behavior_type"].isin([1, 2, 3, 4])
]

print(f"\n异常 behavior_type 行数：{len(invalid_behavior)}")

if len(invalid_behavior) > 0:
    print("异常 behavior_type 示例：")
    print(invalid_behavior.head())

# 将 time 转换为 datetime，非法时间会被转换为 NaT
df["time"] = pd.to_datetime(
    df["time"],
    format="%Y-%m-%d %H",
    errors="coerce",
)

invalid_time = df[df["time"].isna()]
print(f"\n异常时间行数：{len(invalid_time)}")

if len(invalid_time) > 0:
    print("异常时间示例：")
    print(invalid_time.head())

print("\n唯一值数量：")
print(f"user_id 唯一值数量：{df['user_id'].nunique()}")
print(f"item_id 唯一值数量：{df['item_id'].nunique()}")
print(f"item_category 唯一值数量：{df['item_category'].nunique()}")

# 删除关键字段缺失或异常的数据
df_clean = df.dropna(
    subset=[
        "user_id",
        "item_id",
        "behavior_type",
        "item_category",
        "time",
    ]
).copy()

# 只保留有业务意义的用户行为类型
df_clean = df_clean[
    df_clean["behavior_type"].isin([1, 2, 3, 4])
].copy()

# 字段类型转换
df_clean["user_id"] = df_clean["user_id"].astype(str)
df_clean["item_id"] = df_clean["item_id"].astype(str)
df_clean["item_category"] = df_clean["item_category"].astype(str)
df_clean["behavior_type"] = df_clean["behavior_type"].astype(int)

# 字段重命名：time 改成 timestamp，更符合数据库语义
df_clean = df_clean.rename(columns={"time": "timestamp"})

print("\n清洗后数据量：")
print(f"原始数据行数：{len(df)}")
print(f"清洗后数据行数：{len(df_clean)}")
print(f"删除行数：{len(df) - len(df_clean)}")

print("\n清洗后字段信息：")
print(df_clean.info())


# =========================
# 4. 连接 MySQL Server
# =========================

server_engine = create_mysql_server_engine(
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
)

print("\n成功连接 MySQL Server")


# =========================
# 5. 创建 Database
# =========================

create_database(
    server_engine=server_engine,
    database_name=DATABASE_NAME,
)

print(f"数据库 {DATABASE_NAME} 已准备完成")


# =========================
# 6. 连接具体 Database
# =========================

db_engine = create_database_engine(
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    database_name=DATABASE_NAME,
)

print(f"成功连接数据库：{DATABASE_NAME}")
print(f"准备写入记录数：{len(df_clean)}")


# =========================
# 7. 写入 MySQL
# =========================

df_clean.to_sql(
    name="user_behavior",
    con=db_engine,
    if_exists="replace",
    index=False,
)

print("数据已成功写入 MySQL, 表: user_behavior")


# =========================
# 8. SQL 查询验证
# =========================

with db_engine.connect() as conn:
    result = conn.execute(
        text(
            """
            SELECT COUNT(*) AS total_rows
            FROM user_behavior
            """
        )
    )

    for row in result:
        print(f"MySQL 表中数据行数：{row.total_rows}")