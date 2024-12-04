import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

# Load the dataset
file_path = 'twitter_training.csv'
df = pd.read_csv(file_path)

# Rename columns for easier handling
df.columns = ['ID', 'Category', 'Sentiment_Label', 'Text']

# Step 3: Data Cleaning - Remove URLs, mentions, and hashtags from the 'Text' column
df['cleaned_text'] = df['Text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\w+|#', '', str(x)))

# Step 4: Sentiment Analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Step 5: Aggregate sentiment data
sentiment_counts = df['sentiment'].value_counts()

# Step 6: Visualize sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
