from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)
df = heart.data.features.copy()
df["target"] = (heart.data.targets["num"] > 0).astype(int)  # 0 = no disease, 1 = disease
df.to_csv("QA/heart_disease.csv", index=False)
print(df.shape)
print(df.isna().sum())
print(df["target"].value_counts())
