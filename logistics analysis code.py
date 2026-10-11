import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/charts/"
import os; os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------
# 1. DATA SIMULATION
# ------------------------------------------------------------------
N = 2000
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Rail", "Air", "Sea"]
carriers = ["FastTrack", "SwiftHaul", "PrimeLine", "BlueRoute"]

df = pd.DataFrame({
    "shipment_id": [f"SH{10000+i}" for i in range(N)],
    "ship_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D"),
    "region": np.random.choice(regions, N, p=[0.2, 0.25, 0.2, 0.2, 0.15]),
    "mode": np.random.choice(modes, N, p=[0.55, 0.2, 0.15, 0.10]),
    "carrier": np.random.choice(carriers, N),
})

# Distance depends loosely on mode
dist_base = {"Road": 450, "Rail": 900, "Air": 1800, "Sea": 2500}
df["distance_km"] = df["mode"].map(dist_base) * np.random.lognormal(0, 0.35, N)
df["distance_km"] = df["distance_km"].round(1)

# Shipment volume / weight
df["weight_kg"] = np.random.lognormal(6.0, 0.8, N).round(1)
df["volume_m3"] = (df["weight_kg"] / np.random.uniform(150, 300, N)).round(2)

# Speed by mode (km/day) -> transit time
speed = {"Road": 600, "Rail": 450, "Air": 2500, "Sea": 350}
df["transit_days"] = df["distance_km"] / df["mode"].map(speed)

# Warehouse processing delay (bottleneck), worse for peak months and West region
df["month"] = df["ship_date"].dt.month
peak = df["month"].isin([10, 11, 12]).astype(int)
df["warehouse_hours"] = (np.random.gamma(2.0, 3.0, N) + peak * np.random.gamma(2.0, 3.0, N)
                         + (df["region"] == "West") * np.random.gamma(2.0, 2.5, N)).round(1)

# Carrier effect on delay
carrier_delay = {"FastTrack": 0.0, "SwiftHaul": 0.2, "PrimeLine": 0.5, "BlueRoute": 1.1}
df["delivery_days"] = (df["transit_days"] + df["warehouse_hours"] / 24
                       + df["carrier"].map(carrier_delay) * np.random.uniform(0.3, 1.2, N)
                       + np.random.normal(0, 0.3, N)).clip(lower=0.2).round(2)

# Cost model
cost_per_tonne_km = {"Road": 0.12, "Rail": 0.06, "Air": 0.85, "Sea": 0.03}
df["transport_cost"] = (df["weight_kg"] / 1000 * df["distance_km"] * df["mode"].map(cost_per_tonne_km)
                        * np.random.uniform(0.9, 1.15, N) + 40 + df["warehouse_hours"] * 2.5).round(2)
df["cost_per_kg"] = (df["transport_cost"] / df["weight_kg"]).round(3)

# Promised SLA and on-time flag
sla = {"Road": 3, "Rail": 5, "Air": 2, "Sea": 9}
df["sla_days"] = df["mode"].map(sla)
df["on_time"] = (df["delivery_days"] <= df["sla_days"]).astype(int)

df.to_csv("/home/claude/logistics_data.csv", index=False)

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
num_cols = ["distance_km", "weight_kg", "volume_m3", "warehouse_hours",
            "delivery_days", "transport_cost", "cost_per_kg"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["skew"] = df[num_cols].skew()
desc = desc[["mean", "median", "std", "min", "max", "skew"]].round(2)

corr = df[num_cols].corr().round(2)

by_mode = df.groupby("mode").agg(
    shipments=("shipment_id", "count"),
    avg_delivery_days=("delivery_days", "mean"),
    avg_cost=("transport_cost", "mean"),
    avg_cost_per_kg=("cost_per_kg", "mean"),
    on_time_pct=("on_time", lambda s: s.mean() * 100),
).round(2)

by_carrier = df.groupby("carrier").agg(
    shipments=("shipment_id", "count"),
    avg_delivery_days=("delivery_days", "mean"),
    avg_wh_hours=("warehouse_hours", "mean"),
    on_time_pct=("on_time", lambda s: s.mean() * 100),
).round(2)

by_region = df.groupby("region").agg(
    shipments=("shipment_id", "count"),
    avg_wh_hours=("warehouse_hours", "mean"),
    avg_delivery_days=("delivery_days", "mean"),
    on_time_pct=("on_time", lambda s: s.mean() * 100),
).round(2)

monthly = df.groupby("month").agg(
    shipments=("shipment_id", "count"),
    avg_wh_hours=("warehouse_hours", "mean"),
    avg_cost=("transport_cost", "mean"),
    on_time_pct=("on_time", lambda s: s.mean() * 100),
).round(2)

# Cost driver decomposition: cost split
df["linehaul_cost"] = df["transport_cost"] - 40 - df["warehouse_hours"] * 2.5
df["handling_cost"] = df["warehouse_hours"] * 2.5
cost_split = df.groupby("mode")[["linehaul_cost", "handling_cost"]].mean().round(2)
cost_split["fixed"] = 40

overall = {
    "n": N,
    "mean_delivery": round(df.delivery_days.mean(), 2),
    "median_delivery": round(df.delivery_days.median(), 2),
    "overall_on_time": round(df.on_time.mean() * 100, 1),
    "mean_cost": round(df.transport_cost.mean(), 2),
    "median_cost": round(df.transport_cost.median(), 2),
    "peak_wh": round(df[peak == 1].warehouse_hours.mean(), 2),
    "offpeak_wh": round(df[peak == 0].warehouse_hours.mean(), 2),
    "peak_ontime": round(df[peak == 1].on_time.mean() * 100, 1),
    "offpeak_ontime": round(df[peak == 0].on_time.mean() * 100, 1),
    "corr_dist_cost": corr.loc["distance_km", "transport_cost"],
    "corr_wt_cost": corr.loc["weight_kg", "transport_cost"],
    "corr_wh_deliv": corr.loc["warehouse_hours", "delivery_days"],
    "missing": int(df.isna().sum().sum()),
}

tables = {
    "desc": desc.reset_index().rename(columns={"index": "variable"}).values.tolist(),
    "by_mode": by_mode.reset_index().values.tolist(),
    "by_carrier": by_carrier.reset_index().values.tolist(),
    "by_region": by_region.reset_index().values.tolist(),
    "monthly": monthly.reset_index().values.tolist(),
    "corr": corr.reset_index().values.tolist(),
    "corr_cols": list(corr.columns),
    "overall": overall,
    "sample": df.head(5)[["shipment_id", "ship_date", "region", "mode", "carrier",
                          "distance_km", "weight_kg", "warehouse_hours",
                          "delivery_days", "transport_cost", "on_time"]]
              .astype(str).values.tolist(),
}
json.dump(tables, open("/home/claude/tables.json", "w"), default=str)

# ------------------------------------------------------------------
# 3. VISUALIZATIONS
# ------------------------------------------------------------------
# Fig 1: Distribution of delivery time + cost (histograms w/ KDE)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["delivery_days"], bins=40, kde=True, ax=ax[0], color="#2a6f97")
ax[0].axvline(df.delivery_days.mean(), color="red", ls="--", label=f"Mean {df.delivery_days.mean():.1f}")
ax[0].axvline(df.delivery_days.median(), color="green", ls="--", label=f"Median {df.delivery_days.median():.1f}")
ax[0].set_title("Distribution of Delivery Time"); ax[0].set_xlabel("Delivery days"); ax[0].legend()
sns.histplot(df["transport_cost"], bins=40, kde=True, ax=ax[1], color="#e07a5f")
ax[1].axvline(df.transport_cost.mean(), color="red", ls="--", label=f"Mean {df.transport_cost.mean():.0f}")
ax[1].axvline(df.transport_cost.median(), color="green", ls="--", label=f"Median {df.transport_cost.median():.0f}")
ax[1].set_title("Distribution of Transport Cost"); ax[1].set_xlabel("Cost (USD)"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT + "fig1_distributions.png", dpi=150); plt.close()

# Fig 2: Correlation heatmap
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout(); plt.savefig(OUT + "fig2_correlation.png", dpi=150); plt.close()

# Fig 3: Box plot delivery time by mode + cost per kg by mode (log)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(data=df, x="mode", y="delivery_days", order=modes, ax=ax[0], hue="mode", legend=False)
ax[0].set_title("Delivery Time by Transport Mode"); ax[0].set_ylabel("Delivery days")
sns.boxplot(data=df, x="mode", y="cost_per_kg", order=modes, ax=ax[1], hue="mode", legend=False)
ax[1].set_yscale("log"); ax[1].set_title("Cost per kg by Transport Mode (log scale)"); ax[1].set_ylabel("USD / kg")
plt.tight_layout(); plt.savefig(OUT + "fig3_mode_boxplots.png", dpi=150); plt.close()

# Fig 4: Scatter distance vs cost, colored by mode
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="distance_km", y="transport_cost", hue="mode", alpha=0.55, s=25)
plt.yscale("log")
plt.title("Distance vs Transport Cost by Mode (log y-axis)")
plt.xlabel("Distance (km)"); plt.ylabel("Transport cost (USD, log)")
plt.tight_layout(); plt.savefig(OUT + "fig4_distance_cost.png", dpi=150); plt.close()

# Fig 5: Monthly trend (dual panel)
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.bar(monthly.index, monthly["avg_wh_hours"], color="#9ecae1", label="Avg warehouse hours")
ax1.set_xlabel("Month"); ax1.set_ylabel("Avg warehouse hours"); ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["on_time_pct"], color="#d62728", marker="o", label="On-time %")
ax2.set_ylabel("On-time delivery %"); ax2.grid(False)
ax1.set_title("Seasonal Trend: Warehouse Dwell Time vs On-Time Rate")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout(); plt.savefig(OUT + "fig5_monthly_trend.png", dpi=150); plt.close()

# Fig 6: Region x Carrier on-time heatmap
pivot = df.pivot_table(index="region", columns="carrier", values="on_time", aggfunc="mean") * 100
plt.figure(figsize=(7.5, 4.8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=pivot.values.min() - 3, vmax=100,
            cbar_kws={"label": "On-time %"})
plt.title("On-Time Delivery % by Region and Carrier")
plt.tight_layout(); plt.savefig(OUT + "fig6_region_carrier_heatmap.png", dpi=150); plt.close()

# Fig 7: Stacked cost composition by mode
cost_split[["fixed", "handling_cost", "linehaul_cost"]].loc[modes].plot(
    kind="bar", stacked=True, figsize=(8, 4.8), color=["#bdbdbd", "#f4a261", "#2a6f97"])
plt.title("Average Cost Composition per Shipment by Mode")
plt.ylabel("USD per shipment"); plt.xlabel("Mode"); plt.xticks(rotation=0)
plt.legend(["Fixed fee", "Warehouse handling", "Line-haul"])
plt.tight_layout(); plt.savefig(OUT + "fig7_cost_composition.png", dpi=150); plt.close()

# Fig 8: Warehouse hours vs delivery days regression + region violin
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
road = df[df["mode"] == "Road"]
sns.regplot(data=road, x="warehouse_hours", y="delivery_days", scatter_kws={"alpha": .3, "s": 14},
            line_kws={"color": "red"}, ax=ax[0])
ax[0].set_title("Warehouse Hours vs Delivery Days (Road only)")
sns.violinplot(data=df, x="region", y="warehouse_hours", ax=ax[1], hue="region", legend=False, inner="quartile")
ax[1].set_title("Warehouse Hours by Region")
plt.tight_layout(); plt.savefig(OUT + "fig8_bottleneck.png", dpi=150); plt.close()

print(json.dumps(overall, indent=2))
print(desc); print(by_mode); print(by_carrier); print(by_region); print(monthly); print(corr)
print("Road regression slope check:")
print(np.polyfit(road.warehouse_hours, road.delivery_days, 1))
print(pivot.round(1))
print(cost_split)
