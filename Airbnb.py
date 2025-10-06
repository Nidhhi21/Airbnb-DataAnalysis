# =========================
# Airbnb Hotel Booking Analysis - Full Script
# =========================

# 0) Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Pretty display
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
sns.set(style='whitegrid')

# 1) Load Data
# Update this to your dataset location
df = pd.read_csv('Airbnb_Open_Data.csv', low_memory=False)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

print('Raw shape:', df.shape)
print(df.head(3))

# 2) Cleaning
work = df.copy()

# Drop duplicates
work = work.drop_duplicates()

# Drop sparse columns
for col in ['house_rules','license']:
    if col in work.columns and work[col].isna().mean() > 0.9:
        work = work.drop(columns=col)

# Currency cleaning
for col in ['price', 'price_$', 'service_fee', 'service_fee_$']:
    if col in work.columns:
        work[col] = (work[col]
                     .astype(str)
                     .str.replace(r'[\$,]', '', regex=True)
                     .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                     .astype(float))

# Normalize names for price/service_fee
if 'price_$' in work.columns and 'price' not in work.columns:
    work = work.rename(columns={'price_$': 'price'})
if 'service_fee_$' in work.columns and 'service_fee' not in work.columns:
    work = work.rename(columns={'service_fee_$': 'service_fee'})

# Trim text fields
for col in ['neighbourhood_group','neighbourhood','room_type','cancellation_policy','host_identity_verified']:
    if col in work.columns:
        work[col] = work[col].astype(str).str.strip()

# Fix Brooklyn typos
if 'neighbourhood_group' in work.columns:
    work['neighbourhood_group'] = work['neighbourhood_group'].replace({
        'Brookln':'Brooklyn','brookln':'Brooklyn','brooklyn':'Brooklyn'
    })

# Dates and numerics
if 'last_review' in work.columns:
    work['last_review'] = pd.to_datetime(work['last_review'], errors='coerce')

for col in ['minimum_nights','number_of_reviews','reviews_per_month',
            'review_rate_number','calculated_host_listings_count','availability_365','construction_year']:
    if col in work.columns:
        work[col] = pd.to_numeric(work[col], errors='coerce')

# Drop missing core fields
core_cols = [c for c in ['price','room_type','neighbourhood_group','neighbourhood'] if c in work.columns]
work = work.dropna(subset=core_cols, how='any')

# Remove basic outliers
if 'price' in work.columns:
    work = work[(work['price'] >= 10) & (work['price'] <= 2000)]
if 'service_fee' in work.columns:
    work = work[(work['service_fee'] >= 0) & (work['service_fee'] <= 500)]
if 'availability_365' in work.columns:
    work = work[(work['availability_365'] >= 0) & (work['availability_365'] <= 365)]

print('Cleaned shape:', work.shape)

# 3) Q1: Property types present
prop_col = 'property_type' if 'property_type' in work.columns else 'room_type'
q1 = work[prop_col].value_counts().rename_axis(prop_col).reset_index(name='count')
print('\nQ1 - Property types:')
print(q1)

# 4) Q2: Neighborhood group with the highest number of listings
q2 = work['neighbourhood_group'].value_counts().rename_axis('neighbourhood_group').reset_index(name='listings')
print('\nQ2 - Listings by Neighborhood Group:')
print(q2)

# 5) Q3: Neighborhoods with highest average prices
q3 = (work.groupby(['neighbourhood_group','neighbourhood'], as_index=False)['price']
      .mean().sort_values('price', ascending=False).head(10))
print('\nQ3 - Top 10 Neighborhoods by Avg Price:')
print(q3)

# 6) Q4: Relationship between construction year and price
if {'construction_year','price'} <= set(work.columns):
    w_year = work.dropna(subset=['construction_year','price']).copy()
    corr_year_price = w_year[['construction_year','price']].corr().iloc[0,1]
    print('\nQ4 - Corr(construction_year, price):', round(corr_year_price, 3))
    w_year['decade'] = (w_year['construction_year'] // 10) * 10
    q4_decade = w_year.groupby('decade', as_index=False)['price'].median().sort_values('decade')
    print('Median price by decade:')
    print(q4_decade)
else:
    print('\nQ4 - construction_year not available in data.')

# 7) Q5: Top 10 hosts by calculated host listings count
if {'host_id','host_name','calculated_host_listings_count'} <= set(work.columns):
    q5 = (work.groupby(['host_id','host_name'], as_index=False)['calculated_host_listings_count']
          .max().sort_values('calculated_host_listings_count', ascending=False).head(10))
    print('\nQ5 - Top 10 Hosts by Listings Count:')
    print(q5)
else:
    print('\nQ5 - host fields or calculated_host_listings_count missing.')

# 8) Q6: Verified hosts and positive reviews
if 'host_identity_verified' in work.columns and 'review_rate_number' in work.columns:
    q6 = (work.groupby('host_identity_verified')['review_rate_number']
          .agg(['count','mean','median']).reset_index())
    print('\nQ6 - Review rates by Host Identity Verified:')
    print(q6)
else:
    print('\nQ6 - host_identity_verified or review_rate_number missing.')

# 9) Q7: Correlation between price and service fee
if {'price','service_fee'} <= set(work.columns):
    q7_corr = work[['price','service_fee']].corr().iloc[0,1]
    print('\nQ7 - Corr(price, service_fee):', round(q7_corr, 3))
else:
    print('\nQ7 - service_fee not available.')

# 10) Q8: Average review rate by neighborhood group and room type
if {'review_rate_number','neighbourhood_group','room_type'} <= set(work.columns):
    q8 = (work.groupby(['neighbourhood_group','room_type'], as_index=False)['review_rate_number']
          .mean().sort_values(['neighbourhood_group','review_rate_number'], ascending=[True, False]))
    print('\nQ8 - Avg Review Rate by Group & Room Type:')
    print(q8)
else:
    print('\nQ8 - columns missing for this question.')

# 11) Q9: Do hosts with more listings keep higher availability?
if {'calculated_host_listings_count','availability_365'} <= set(work.columns):
    q9_corr = work[['calculated_host_listings_count','availability_365']].corr().iloc[0,1]
    print('\nQ9 - Corr(host listings count, availability_365):', round(q9_corr, 3))

    bins = pd.cut(
        work['calculated_host_listings_count'],
        bins=[0,1,3,10,100, work['calculated_host_listings_count'].max()],
        include_lowest=True
    )
    q9_bins = work.groupby(bins, as_index=False)['availability_365'].median()
    q9_bins = q9_bins.rename(columns={'calculated_host_listings_count':'host_size_bin',
                                      'availability_365':'median_availability_365'})
    print('Median availability by host-size bin:')
    print(q9_bins)
else:
    print('\nQ9 - columns missing for this question.')

# 12) Visualizations (export-ready)
# Create an output folder for images
out_dir = Path('airbnb_outputs')
out_dir.mkdir(exist_ok=True)

# Plot: Listings by Neighbourhood Group
plt.figure(figsize=(8,4))
sns.countplot(data=work, x='neighbourhood_group',
              order=work['neighbourhood_group'].value_counts().index, palette='Blues_r')
plt.title('Listings by Neighborhood Group')
plt.xlabel('Neighborhood Group'); plt.ylabel('Listings')
plt.tight_layout(); plt.savefig(out_dir/'listings_by_group.png', dpi=200); plt.show()

# Plot: Top 10 Neighborhoods by Mean Price
topN = (work.groupby('neighbourhood', as_index=False)['price'].mean()
        .sort_values('price', ascending=False).head(10))
plt.figure(figsize=(8,5))
sns.barplot(data=topN, y='neighbourhood', x='price', palette='Reds_r')
plt.title('Top 10 Neighborhoods by Mean Price')
plt.xlabel('Average Price'); plt.ylabel('Neighborhood')
plt.tight_layout(); plt.savefig(out_dir/'top10_neighborhoods_price.png', dpi=200); plt.show()

# Plot: Price vs Service Fee
if {'price','service_fee'} <= set(work.columns):
    sample = work[['price','service_fee']].dropna().sample(min(8000, len(work)), random_state=42)
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=sample, x='service_fee', y='price', alpha=0.3)
    plt.title('Price vs Service Fee')
    plt.tight_layout(); plt.savefig(out_dir/'price_vs_service_fee.png', dpi=200); plt.show()

# Plot: Price vs Construction Year with decade trend
if {'construction_year','price'} <= set(work.columns):
    w_year = work.dropna(subset=['construction_year','price'])
    samp = w_year.sample(min(6000, len(w_year)), random_state=0)
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=samp, x='construction_year', y='price', alpha=0.25)
    w_year['decade'] = (w_year['construction_year'] // 10) * 10
    decade_price = w_year.groupby('decade', as_index=False)['price'].median().sort_values('decade')
    sns.lineplot(data=decade_price, x='decade', y='price', color='red')
    plt.title('Price vs Construction Year')
    plt.tight_layout(); plt.savefig(out_dir/'price_vs_construction_year.png', dpi=200); plt.show()

print('\nAll done! Figures saved to:', out_dir.resolve())
