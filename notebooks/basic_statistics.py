# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Basic statistics: how do transaction amounts vary across channels?
# MAGIC
# MAGIC **Student:** Emilio Alvarado Carson (A01751864) · TI2006B · Activity 4.2, MLOps in Databricks
# MAGIC
# MAGIC | Item | Value |
# MAGIC |---|---|
# MAGIC | **Question** | How do transaction amounts vary across channels (Online vs Store)? |
# MAGIC | **Data source** | Synthetic example from the activity instructions, generated in this notebook with NumPy (`default_rng(42)`). No external, private or personal data. |
# MAGIC | **Numeric column** | `amount`: transaction amount in *synthetic units* (no real currency) |
# MAGIC | **Category** | `channel`: `Online` or `Store` (100 rows each) |
# MAGIC | **Missing-value policy** | One amount is set to missing on purpose (row 0). Rows with a missing `amount` or `channel` are **excluded** from the analysis, not filled in. |
# MAGIC | **Filtering / sampling** | None besides removing the missing row. All 200 generated rows are used. |
# MAGIC
# MAGIC **Expected counts:** 200 raw rows, 1 missing amount, 199 analyzed rows. The grouped counts must add up to 199.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment
# MAGIC pandas and NumPy come with the Databricks environment. Plotly is installed here at notebook scope so the interactive run and the Job use the same library. Python restarts after `%pip`, so this cell goes first.

# COMMAND ----------

# MAGIC %pip install -q plotly

# COMMAND ----------

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

print("Versions:", np.__version__, pd.__version__, plotly.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the data and check missing values

# COMMAND ----------

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "channel": ["Online"] * 100 + ["Store"] * 100,
    "amount": np.round(rng.lognormal(3.5, 0.6, 200), 2),
})
df.loc[0, "amount"] = np.nan

print("Rows:", len(df))
print("Missing values:", df[["amount", "channel"]].isna().sum())
clean = df.dropna(subset=["amount", "channel"]).copy()
if len(clean) < 2 or not np.isfinite(clean["amount"]).all():
    raise ValueError("Need at least two finite amounts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Descriptive statistics

# COMMAND ----------

print(clean["amount"].describe().round(2))
print("Median:", round(clean["amount"].median(), 2))
by_channel = clean.groupby("channel")["amount"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
print(by_channel)

# Count reconciliation: raw rows = analyzed rows + excluded rows, and the groups add up to the analyzed rows
excluded = len(df) - len(clean)
assert len(df) == 200 and excluded == 1 and len(clean) == 199
assert by_channel["count"].sum() == len(clean)
print(f"Raw rows: {len(df)} | excluded (missing amount): {excluded} | analyzed: {len(clean)} | sum of group counts: {by_channel['count'].sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary of the statistics (seed 42)
# MAGIC
# MAGIC | Statistic | All channels | Online | Store |
# MAGIC |---|---:|---:|---:|
# MAGIC | Rows analyzed | 199 | 99 | 100 |
# MAGIC | Mean | 37.61 | 35.65 | 39.55 |
# MAGIC | Median | 32.06 | 32.78 | 30.64 |
# MAGIC | Sample standard deviation | 23.12 | 17.01 | 27.84 |
# MAGIC | Minimum | 9.21 | 10.27 | 9.21 |
# MAGIC | Q1 (25%) | 22.31 | | |
# MAGIC | Q3 (75%) | 45.72 | | |
# MAGIC | Maximum | 190.25 | 119.70 | 190.25 |
# MAGIC
# MAGIC Total rows: **200**. Missing values: **1** in `amount`, **0** in `channel`. Analyzed rows: **199**. Online 99 + Store 100 = 199, so the counts reconcile.
# MAGIC
# MAGIC **Why one row was excluded:** row 0 (an Online transaction) has no amount, and filling it with zero or with the mean would invent a value. Excluding it leaves Online with 99 rows and Store with 100. With 199 of 200 rows the effect on the statistics is minimal. With real data, though, missing values are rarely random: if, for example, large online purchases failed to record their amount more often, dropping them would bias the Online mean downward.
# MAGIC
# MAGIC *pandas `std` is the sample standard deviation (n − 1 in the denominator).*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interactive Plotly charts

# COMMAND ----------

histogram = px.histogram(
    clean, x="amount", nbins=20, title="Transaction amount distribution",
    labels={"amount": "Amount (synthetic units)"},
)
histogram.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Histogram interpretation.** The distribution is **right-skewed** (skewness ≈ 2.4): most transactions fall between roughly 10 and 50 units, and a long tail reaches up to 190.25. Because of that tail, the **mean (37.61) is higher than the median (32.06)**, and only about 40% of transactions are above the mean. For a "typical" transaction, the median is the better summary here. The spread is also wide: the standard deviation is 23.12, and half of the transactions fall between Q1 = 22.31 and Q3 = 45.72.

# COMMAND ----------

box = px.box(
    clean, x="channel", y="amount", title="Amounts by channel",
    labels={"amount": "Amount (synthetic units)", "channel": "Channel"},
)
box.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Box plot interpretation.** The medians of both channels are very close (Online 32.78, Store 30.64), and the boxes overlap almost completely, so a typical transaction is similar in both channels. The difference is in the tail: **Store has more and larger high values** (9 points above its upper whisker against 2 for Online, including the maximum of 190.25). That is why its standard deviation (27.84 against 17.01) and its mean (39.55 against 35.65) are higher, even though its median is *lower*.
# MAGIC
# MAGIC **Observation supported by the statistics:** the higher Store mean comes from a few large transactions, not from typical ones, as the higher Store standard deviation and lower Store median show.
# MAGIC **Observation supported by the plot:** the box plot shows the extra high outliers in Store, and the almost identical boxes and medians.
# MAGIC
# MAGIC **Uneven group sizes:** Online has 99 rows and Store 100, because the excluded row was Online. The difference is too small to affect the comparison.
# MAGIC
# MAGIC **Limitation:** both channels were generated from **the same distribution** (lognormal, μ = 3.5, σ = 0.6), so there is no real channel effect. The differences we see are random sampling variation from 100 rows per group. No statistical test was run, and with another seed Online could just as easily have the higher mean. These conclusions describe this synthetic sample, not real transactions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save a run summary to a Delta table (used by the Databricks alert)
# MAGIC
# MAGIC Each run (interactive or Job) appends one row with its counts and main statistics to `workspace.default.basic_statistics_summary`. A Databricks SQL alert reads the latest row and notifies when the data has missing values. This is how the Job's results can be checked without opening the notebook.

# COMMAND ----------

from datetime import datetime, timezone

summary = pd.DataFrame([{
    "run_ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "raw_rows": int(len(df)),
    "missing_amount": int(df["amount"].isna().sum()),
    "analyzed_rows": int(len(clean)),
    "mean_amount": float(round(clean["amount"].mean(), 2)),
    "median_amount": float(round(clean["amount"].median(), 2)),
    "std_amount": float(round(clean["amount"].std(), 2)),
}])
SUMMARY_TABLE = "workspace.default.basic_statistics_summary"
spark.createDataFrame(summary).write.format("delta").mode("append").saveAsTable(SUMMARY_TABLE)
display(spark.table(SUMMARY_TABLE).orderBy("run_ts_utc", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conclusions
# MAGIC - Transaction amounts are right-skewed: the median (32.06) describes a typical transaction better than the mean (37.61).
# MAGIC - Online and Store have almost the same typical amount. Store shows more large transactions, which raise its mean and its spread.
# MAGIC - Since both channels come from the same synthetic distribution, these differences are sampling noise and should not be read as a channel effect.
# MAGIC
# MAGIC **Reproduction:** run on Databricks serverless with pandas, NumPy and Plotly (versions printed in section 1). Seed `42`; the only filter is the exclusion of rows with a missing amount or channel. Run all cells in order from a clean session.