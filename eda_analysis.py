import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("netflix_data.csv")

# Display first rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna("Unknown", inplace=True)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include='all'))

# Count Movies vs TV Shows
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.savefig("movies_vs_tvshows.png")
plt.show()

# Top 10 countries producing content
plt.figure(figsize=(10,5))

top_countries = df['country'].value_counts().head(10)

sns.barplot(x=top_countries.index, y=top_countries.values)

plt.xticks(rotation=45)
plt.title("Top 10 Content Producing Countries")
plt.ylabel("Count")
plt.savefig("top_countries.png")
plt.show()

# Content release trend
plt.figure(figsize=(10,5))

release_year = df['release_year'].value_counts().sort_index()

plt.plot(release_year.index, release_year.values)

plt.title("Content Release Trend")
plt.xlabel("Year")
plt.ylabel("Number of Releases")
plt.savefig("release_trend.png")
plt.show()

# Ratings distribution
plt.figure(figsize=(8,5))

sns.countplot(y='rating', data=df,
              order=df['rating'].value_counts().index)

plt.title("Content Ratings Distribution")
plt.savefig("ratings_distribution.png")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

if not numeric_df.empty:
    plt.figure(figsize=(6,4))
    sns.heatmap(numeric_df.corr(),
                annot=True,
                cmap='coolwarm')

    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()

print("\nEDA Completed Successfully")