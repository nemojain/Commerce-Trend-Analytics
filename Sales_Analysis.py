import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#30363d'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

ACCENT = '#58a6ff'
ACCENT2 = '#3fb950'
ACCENT3 = '#f78166'
ACCENT4 = '#d2a8ff'

df = pd.read_csv('Sales_Data.csv', encoding='latin1')

# ─────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────
print("=" * 50)
print("DATA CLEANING")
print("=" * 50)
print(f"Original shape: {df.shape}")
df.dropna(subset=['Amount'], inplace=True)
print(f"After dropping null amounts: {df.shape}")
print(f"Missing values remaining:\n{df.isnull().sum()}")

# ─────────────────────────────────────────
# KEY METRICS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY METRICS")
print("=" * 50)
total_revenue = df['Amount'].sum()
total_orders = df['Orders'].sum()
total_customers = df['User_ID'].nunique()
avg_order_value = total_revenue / total_orders
print(f"Total Revenue:        ₹{total_revenue:,.0f}")
print(f"Total Orders:         {total_orders:,.0f}")
print(f"Unique Customers:     {total_customers:,.0f}")
print(f"Avg Order Value:      ₹{avg_order_value:,.2f}")

# ─────────────────────────────────────────
# CHART 1: Revenue by State (Top 10)
# ─────────────────────────────────────────
state_revenue = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(state_revenue.index[::-1], state_revenue.values[::-1], color=ACCENT, edgecolor='none')
ax.set_title('Top 10 States by Revenue', fontsize=16, fontweight='bold', pad=15, color='white')
ax.set_xlabel('Total Revenue (₹)', fontsize=12)
for bar, val in zip(bars, state_revenue.values[::-1]):
    ax.text(bar.get_width() + total_revenue * 0.002, bar.get_y() + bar.get_height()/2,
            f'₹{val/1e6:.1f}M', va='center', fontsize=9, color='white')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e6:.0f}M'))
plt.tight_layout()
plt.savefig('chart1_revenue_by_state.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: chart1_revenue_by_state.png")

# ─────────────────────────────────────────
# CHART 2: Revenue by Age Group & Gender
# ─────────────────────────────────────────
age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
age_gender = df.groupby(['Age Group', 'Gender'])['Amount'].sum().unstack()
age_gender = age_gender.reindex(age_order)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(age_gender.index))
width = 0.35
bars1 = ax.bar(x - width/2, age_gender.get('F', 0), width, label='Female', color=ACCENT4, edgecolor='none')
bars2 = ax.bar(x + width/2, age_gender.get('M', 0), width, label='Male', color=ACCENT, edgecolor='none')
ax.set_title('Revenue by Age Group & Gender', fontsize=16, fontweight='bold', pad=15, color='white')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Total Revenue (₹)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(age_gender.index)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e6:.0f}M'))
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
plt.tight_layout()
plt.savefig('chart2_age_gender_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: chart2_age_gender_revenue.png")

# ─────────────────────────────────────────
# CHART 3: Top 10 Product Categories
# ─────────────────────────────────────────
cat_revenue = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
colors = [ACCENT if i < 3 else '#30363d' for i in range(len(cat_revenue))]
bars = ax.bar(range(len(cat_revenue)), cat_revenue.values, color=colors, edgecolor='none')
ax.set_title('Top 10 Product Categories by Revenue', fontsize=16, fontweight='bold', pad=15, color='white')
ax.set_ylabel('Total Revenue (₹)', fontsize=12)
ax.set_xticks(range(len(cat_revenue)))
ax.set_xticklabels(cat_revenue.index, rotation=35, ha='right', fontsize=9)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e6:.0f}M'))
for bar, val in zip(bars, cat_revenue.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_revenue * 0.001,
            f'₹{val/1e6:.1f}M', ha='center', fontsize=8, color='white')
plt.tight_layout()
plt.savefig('chart3_product_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: chart3_product_categories.png")

# ─────────────────────────────────────────
# CHART 4: Revenue by Zone
# ─────────────────────────────────────────
zone_revenue = df.groupby('Zone')['Amount'].sum().sort_values(ascending=False)
zone_pct = zone_revenue / zone_revenue.sum() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
colors_zone = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#ffa657']
ax1.bar(zone_revenue.index, zone_revenue.values, color=colors_zone, edgecolor='none')
ax1.set_title('Revenue by Zone', fontsize=14, fontweight='bold', color='white')
ax1.set_ylabel('Total Revenue (₹)', fontsize=11)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e6:.0f}M'))

# Pie chart
wedges, texts, autotexts = ax2.pie(
    zone_revenue.values,
    labels=zone_revenue.index,
    autopct='%1.1f%%',
    colors=colors_zone,
    startangle=90,
    wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
)
for text in texts:
    text.set_color('white')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
ax2.set_title('Zone Revenue Share', fontsize=14, fontweight='bold', color='white')

plt.suptitle('Geographic Revenue Distribution', fontsize=16, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('chart4_zone_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: chart4_zone_revenue.png")

# ─────────────────────────────────────────
# CHART 5: Top Occupations by Revenue
# ─────────────────────────────────────────
occ_revenue = df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False).head(8)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(occ_revenue.index[::-1], occ_revenue.values[::-1],
               color=[ACCENT2 if i < 3 else ACCENT for i in range(len(occ_revenue))], edgecolor='none')
ax.set_title('Top Occupations by Revenue', fontsize=16, fontweight='bold', pad=15, color='white')
ax.set_xlabel('Total Revenue (₹)', fontsize=12)
for bar, val in zip(bars, occ_revenue.values[::-1]):
    ax.text(bar.get_width() + total_revenue * 0.001, bar.get_y() + bar.get_height()/2,
            f'₹{val/1e6:.1f}M', va='center', fontsize=9, color='white')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e6:.0f}M'))
plt.tight_layout()
plt.savefig('chart5_occupation_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: chart5_occupation_revenue.png")

# ─────────────────────────────────────────
# CHART 6: Marital Status & Gender Split
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gender
gender_rev = df.groupby('Gender')['Amount'].sum()
gender_labels = ['Female' if g == 'F' else 'Male' for g in gender_rev.index]
axes[0].pie(gender_rev.values, labels=gender_labels, autopct='%1.1f%%',
            colors=[ACCENT4, ACCENT], startangle=90,
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
for text in axes[0].texts:
    text.set_color('white')
axes[0].set_title('Revenue by Gender', fontsize=13, fontweight='bold', color='white')

# Marital Status
marital_rev = df.groupby('Marital_Status')['Amount'].sum()
marital_labels = ['Married' if m == 1 else 'Single' for m in marital_rev.index]
axes[1].pie(marital_rev.values, labels=marital_labels, autopct='%1.1f%%',
            colors=[ACCENT2, ACCENT3], startangle=90,
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
for text in axes[1].texts:
    text.set_color('white')
axes[1].set_title('Revenue by Marital Status', fontsize=13, fontweight='bold', color='white')

plt.suptitle('Customer Demographics', fontsize=16, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('chart6_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: chart6_demographics.png")

# ─────────────────────────────────────────
# KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
top_state = state_revenue.index[0]
top_state_rev = state_revenue.values[0]
top_category = cat_revenue.index[0]
top_zone = zone_revenue.index[0]
top_age = df.groupby('Age Group')['Amount'].sum().idxmax()
top_occupation = occ_revenue.index[0]
female_rev = df[df['Gender'] == 'F']['Amount'].sum()
male_rev = df[df['Gender'] == 'M']['Amount'].sum()

print(f"1. Top State:         {top_state} (₹{top_state_rev/1e6:.1f}M)")
print(f"2. Top Category:      {top_category}")
print(f"3. Top Zone:          {top_zone}")
print(f"4. Highest Spending Age Group: {top_age}")
print(f"5. Top Occupation:    {top_occupation}")
print(f"6. Female Revenue:    ₹{female_rev/1e6:.1f}M ({female_rev/total_revenue*100:.1f}%)")
print(f"7. Male Revenue:      ₹{male_rev/1e6:.1f}M ({male_rev/total_revenue*100:.1f}%)")
print("\n✅ All charts saved successfully!")
print("📁 Upload all .png files + this script to GitHub")
