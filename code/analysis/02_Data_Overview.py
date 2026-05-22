import pandas as pd


# 1. 读取清洗后的 CSV
csv_path = "data/processed/Clean_User_Behavior.csv"

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])


# 2. 数据总体概览
print("\n===== 数据总体概览 =====")
print("总记录数量：", len(df))
print("总用户数量：", df["user_id"].nunique())
print("总商品数量：", df["item_id"].nunique())
print("总类目数量：", df["item_category"].nunique())

print("\n===== 时间范围 =====")
print("最早记录用户行为时间：", df["timestamp"].min())
print("最晚记录用户行为时间：", df["timestamp"].max())

print("\n===== 用户平均行为 =====")
print(
    "平均每用户对商品产生行为的数量：",
    len(df) / df["user_id"].nunique()
)