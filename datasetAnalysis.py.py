import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'Global_Music_Streaming_Listener_Preferences.csv'
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')

# =========================
# EXPLORATORY DATA ANALYSIS (EDA) + CLEANING + FEATURE ENGINEERING
# =========================

# Preview the first few rows
print("Dataset Preview:")
print(df.head(), "\n")

# Basic information
print("Dataset Info:")
print(df.info(), "\n")

# Check for missing values
print("Missing Values (Before Cleaning):")
print(df.isnull().sum(), "\n")

# Fill missing numerical columns with median
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Confirm missing values handled
print("Missing Values (After Cleaning):")
print(df.isnull().sum(), "\n")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering
# -------------------------

# Create 'High_Engagement' flag based on engagement rate
df['High_Engagement'] = df['Discover_Weekly_Engagement_'] > 70

# Normalize streaming minutes per day for consistency (optional)
df['Stream_Minutes_Normalized'] = (
    df['Minutes_Streamed_Per_Day'] - df['Minutes_Streamed_Per_Day'].mean()
) / df['Minutes_Streamed_Per_Day'].std()

# Create binary flag for whether user likes more than 50 songs
df['Heavy_Liker'] = df['Number_of_Songs_Liked'] > 50

#Age Segmentation
bins = [10, 18, 25, 35, 50, 65]
labels = ['Teen (13-18)', 'Young Adult (19-25)', 'Adult (26-35)', 'Mid-age (36-50)', 'Senior (51+)']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Summary statistics
print("Summary Statistics (Numerical):")
print(df.describe(), "\n")

print("Summary (Categorical):")
print(df.select_dtypes(include='object').describe(), "\n")

# Unique values in each column
print("Unique Values in Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
print("\n")

# Distribution plots for selected numerical columns
selected_cols = ['Minutes_Streamed_Per_Day', 'Number_of_Songs_Liked', 'Discover_Weekly_Engagement_', 'Repeat_Song_Rate_']
for col in selected_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='steelblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# =========================
# OBJECTIVE 1: Popular Platforms & Genres
# =========================
plt.figure(figsize=(12, 5))

# Streaming Platform
plt.subplot(1, 2, 1)
sns.countplot(
    data=df,
    y='Streaming_Platform',
    order=df['Streaming_Platform'].value_counts().index,
    hue='Streaming_Platform',
    palette='viridis',
    legend=False
)
plt.title('Most Popular Streaming Platforms')
plt.xlabel('User Count')
plt.ylabel('')

# Top Genres
plt.subplot(1, 2, 2)
sns.countplot(
    data=df,
    y='Top_Genre',
    order=df['Top_Genre'].value_counts().index,
    hue='Top_Genre',
    palette='magma',
    legend=False
)
plt.title('Most Popular Music Genres')
plt.xlabel('User Count')
plt.ylabel('')

plt.tight_layout()
plt.show()

# =========================
# OBJECTIVE 2: Listening Habits by Age Group
# =========================
bins = [10, 18, 25, 35, 50, 65]
labels = ['Teen (13-18)', 'Young Adult (19-25)', 'Adult (26-35)', 'Mid-age (36-50)', 'Senior (51+)']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x='Age_Group',
    hue='Listening_Time_MorningAfternoonNight',
    palette='Set2'
)
plt.title('Preferred Listening Time by Age Group')
plt.xlabel('Age Group')
plt.ylabel('User Count')
plt.legend(title='Listening Time')
plt.tight_layout()
plt.show()

# =========================
# OBJECTIVE 3: Engagement by Subscription Type
# =========================
plt.figure(figsize=(12, 5))

# Discover Weekly Engagement
plt.subplot(1, 2, 1)
sns.boxplot(
    data=df,
    x='Subscription_Type',
    y='Discover_Weekly_Engagement_',
    hue='Subscription_Type',
    palette='coolwarm',
    legend=False
)
plt.title('Discover Weekly Engagement by Subscription Type')
plt.ylabel('Engagement (%)')

# Repeat Song Rate
plt.subplot(1, 2, 2)
sns.boxplot(
    data=df,
    x='Subscription_Type',
    y='Repeat_Song_Rate_',
    hue='Subscription_Type',
    palette='coolwarm',
    legend=False
)
plt.title('Repeat Song Rate by Subscription Type')
plt.ylabel('Repeat Rate (%)')

plt.tight_layout()
plt.show()

# =========================
# OBJECTIVE 4: Outlier Detection
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=df,
    y='Minutes_Streamed_Per_Day',
    ax=axes[0],
    color='skyblue'
)
axes[0].set_title('Outliers in Minutes Streamed Per Day')

sns.boxplot(
    data=df,
    y='Number_of_Songs_Liked',
    ax=axes[1],
    color='salmon'
)
axes[1].set_title('Outliers in Number of Songs Liked')

plt.tight_layout()
plt.show()

# =========================
# OBJECTIVE 5: Country-wise Genre Preferences (Top 5 Countries)
# =========================
top_countries = df['Country'].value_counts().nlargest(5).index
df_top_countries = df[df['Country'].isin(top_countries)]

plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_top_countries,
    x='Country',
    hue='Top_Genre',
    palette='tab20'
)
plt.title('Top Genres by Country (Top 5 Countries)')
plt.ylabel('User Count')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
