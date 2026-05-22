from pathlib import Path
import pandas as pd


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
duplicate_ratio = duplicate_count / len(df)

print(f"\n完全重复行数量：{duplicate_count}")
print(f"完全重复行比例：{duplicate_ratio:.2%}")

print("\nbehavior_type 分布：")
print(df["behavior_type"].value_counts(dropna=False))


# =========================
# 3. 数据清洗与字段转换
# =========================

# 用户行为字段 behavior_type 应只包含：
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

# 字段重命名：time 改成 timestamp，更符合后续分析语义
df_clean = df_clean.rename(columns={"time": "timestamp"})

print("\n清洗后数据量：")
print(f"原始数据行数：{len(df)}")
print(f"清洗后数据行数：{len(df_clean)}")
print(f"删除行数：{len(df) - len(df_clean)}")

print("\n清洗后字段信息：")
print(df_clean.info())


# =========================
# 4. 保存清洗后的数据
# =========================

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

clean_csv_path = processed_dir / "clean_user_behavior.csv"

df_clean.to_csv(
    clean_csv_path,
    index=False,
)

print(f"\n清洗后的数据已保存至：{clean_csv_path}")