# FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis
This project analyzes customer sentiments and reviews for the Red Roses Bouquet sold on FlowerAura, an online flower and gifting platform.

As a Data Analyst at FlowerAura, the goal is to understand customer perceptions by collecting, cleaning, and analyzing user-posted reviews from the website. Using BeautifulSoup, Pandas, and TextBlob, the analysis identifies whether customers express positive or negative emotions about their purchase experience.

## Objectives

1. Scrape reviews of the Red Roses Bouquet from FlowerAura’s product pages.
2. Clean and preprocess review data for sentiment analysis.
3. Use TextBlob to compute sentiment polarity and classify reviews as positive or negative.
4. Visualize results through graphs and word clouds to highlight customer opinions and recurring themes.
5. Summarize findings and key recommendations for business improvement.

## Tools and Libraries

1. BeautifulSoup – Web scraping reviews
2. Requests – Fetching page content
3. Pandas – Data handling and analysis
4. TextBlob – Sentiment analysis
5. Matplotlib / Seaborn – Visualization
6. WordCloud – Keyword insights

## Data Collection

### Import all the library that is needed.

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
```
### Empty Lists that will be used to store the scraped data.

```python
Names = []
Ratings = []
Reviews = []
Cities = []
PostedOn = []
Occassions = []
```

### Scraping Customer Reviews from Floweraura Using Requests and BeautifulSoup

```python
url = "https://www.floweraura.com/reviews/p/6617/10-red-roses-bouquet?page="

for i in range(1,51):
    cnp = url+str(i)
    url_new = cnp
    r = requests.get(url_new)

    soup = BeautifulSoup (r.text, "html.parser")
    
    main = soup.find("div", {"class":"review-left-container"})
    
    sub = main.find_all ("div", {"class":"new-review-card-container"})
    
    for i in sub:

        #scrape Name of the Reviewer
        name= i.find("span", {"class":"review-author-name"})
        Names.append(name.text.title())
    
        #scraped City of the Reviewer
        city = i.find_all ("span", {"class":"review-meta-details"})
        Cities.append(city[0].text.title())
        
        try:
            Occassions.append(city[1].text.title())
        except: 
            Occassions.append(np.nan)
    
        #scraped Date of the Reviewing
        date = i.find_all("span")
        try:
            PostedOn.append(date[4]. text)
        except:
            PostedOn.append(nan)

        #scraped Ratings of the Reviewing
        rating = i.find("span", {"class":"star-count-container"})
        Ratings.append(rating.text)
    
         #scraped Reviews Written by the Reviewer
        review = i.find_all("div")
        Reviews.append(review[-1].text)
```

### Creating a DataFrame of Scraped Floweraura Reviews

```python
df = pd.DataFrame({'Names':Names , 'Cities':Cities , 'Posted_On':PostedOn , 'Occasions':Occassions , 'Rating': Ratings , 'Reviews': Reviews })
df
 ```       
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/DataFrame%20of%20Scraped%20Floweraura%20Reviews.jpg)

## Data Cleaning and Preprocessing

### Extracting and Cleaning from Posted_On and Occassions Columns

```python
def extract(value):
    try:
        x=value.index(':')
        return value[x+2:]
    except:
        return np.nan

df['Posted_On'] = df['Posted_On'].apply(extract)
df['Occasions'] = df['Occasions'].apply(extract)
```

### Removing (th, rd,st,nd) from Posted_On Columns
```python
rep = ['th', 'rd', 'st', 'nd']
for i in rep:
    df['Posted_On'] = df['Posted_On'].str.replace(i, "")
```

### Checking the datatype of each.
```python
df.info()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Dataframe%20datatype.jpg)

### Changing Posted_On to Datetime & "Rating Datatype" to int.
```python
df['Posted_On'] = pd.to_datetime(df['Posted_On'])
df['Rating'] = df['Posted_On'].astype("int")
```

## Sentiment Analysis

### Polarity Score
```python
df['Polarity'] = [TextBlob(i).sentiment.polarity for i in df['Reviews']]
df['Polarity'] = df['Polarity'].round(2)
```

### Subjectivity Score
```python
df['Subjectivity'] = [TextBlob(i).sentiment.subjectivity for i in df['Reviews']]
df['Subjectivity'] = df['Subjectivity'].round(2)
```


## Data Visualization & Insights

```python
def score (value):
    if value <= -0.3:
        return "Negative"
    else:
        return "Positive"
df['Score'] = df ["Polarity"].apply(score)
```

### Plots figure for Sentiment Distribution based on Sentiment Category
```python
ax = sns.countplot(x=df['Score'], data = df, color='orange')
ax.bar_label(container = ax.containers[0])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')
plt.show()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Sentiment%20Distribution.jpg)

### Visualizing Positive Customer Reviews Using WordCloud

```python
df_pos = df.loc[df["Score"] == "Positive"]
all_text = " ".join(text for text in df_pos["Reviews"])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.title('Positive Reviews')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Positive%20Customer%20Reviews%20Using%20WordCloud.jpg)

### Visualizing Negative Customer Reviews Using WordCloud
```python
df_neg = df.loc[df["Score"] == "Negative"]
all_text = " ".join(text for text in df_neg["Reviews"])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.title('Negative Reviews')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Negative%20Customer%20Reviews%20Using%20WordCloud.jpg)

### Average Rating vs Sentiment Polarity

```python
# Group properly by rating
rating_sentiment = df.groupby('Rating')['Polarity'].mean().reset_index()
rating_sentiment['Polarity'] = rating_sentiment['Polarity'].round(2)

plt.figure(figsize=(12,6))
sns.boxplot(data=rating_sentiment, x='Polarity', y='Rating', hue = 'Polarity' ,palette='coolwarm')
plt.title('Rating vs Average Polarity')
plt.xlabel('Average Polarity')
plt.xticks(rotation=90)
plt.ylabel(' Rating (1-5 stars) ')
plt.show()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Rating%20vs%20Average%20Polarity.jpg)

## Review Length vs Sentiment Polarity
```python
df['review_length'] = df['Reviews'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12,5))
sns.boxplot(data=df, x='review_length', y='Polarity', hue = 'Score', palette='Set2')
plt.title('Review Length vs Sentiment Polarity')
plt.xlabel('Review Length (Word Count)')
plt.ylabel('Sentiment Polarity')
plt.show()
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Review%20Length%20vs%20Sentiment%20Polarity.jpg)


### Correlation between Review Length and Sentiment Polarity
```python
length_correlation = df['review_length'].corr(df['Polarity'])
print(f"Correlation between Review Length and Sentiment Polarity: {length_correlation:.2f}")
```
![Dashboard Screenshot](https://github.com/shubh-verma96/FlowerAura-Red-Roses-Bouquet-Customer-Sentiment-Analysis/blob/main/Correlation%20Coefficient.jpg)

1. The Correlation value is -0.14 - The minus sign means when review length increases, the positivity slightly decreases.
2. But the number is small (0.14) - That means the connection is very weak.
3. So overall - Longer reviews are a little less positive, but not by much — review length doesn’t really affect sentiment much.

## FlowerAura Reviews Analysis

### Overview:

Reviews were scraped from FlowerAura, cleaned, and analyzed using TextBlob for sentiment classification: extremely positive, positive, neutral, negative, or extremely negative.

### Results:

Most reviews were positive, highlighting high customer satisfaction. Negative/neutral reviews mostly mentioned delivery delays or packaging issues.

### Insights:

Positive: Fast delivery, fresh flowers, good service, attractive bouquets. Common words: "Good," "Thank," "fresh," "beautiful," "service".
Issues: Late deliveries, weather delays, occasional poor flower quality. Common words: "bad," "weather," "despite," "delivering".

### Recommendations:

1. Ensure timely delivery.
2. Prepare for weather-related delays.
3. Improve flower quality checks.
4. Communicate delays clearly.
5. Strengthen customer support.
6. Offer loyalty rewards.




