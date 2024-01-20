import os
import json
import pandas as pd
from datetime import datetime


def get_news(tickers):
    # If tickers is a string, convert it to a list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Initialize an empty list to hold the data
    data = []

    # Define the set of topics
    topics = {'Financial Markets', 'Blockchain', 'Finance', 'IPO', 'Economy - Macro', 'Economy - Fiscal', 'Mergers & Acquisitions', 'Technology', 'Retail & Wholesale', 'Manufacturing', 'Economy - Monetary', 'Earnings', 'Real Estate & Construction', 'Energy & Transportation', 'Life Sciences'}

    # Loop through each ticker
    for ticker in tickers:

        # Try to open the file for the ticker
        try:
            with open(f'data/news/json/{ticker}.json', 'r') as f:
                # Load the JSON data
                json_data = json.load(f)
                
                # Loop through each article in the 'feed' list
                for article in json_data['feed']:
                    # Extract the required fields
                    title = article.get('title')
                    url = article.get('url')

                    # Convert the time_published string to a datetime object
                    time_published = datetime.strptime(article.get('time_published'), '%Y%m%dT%H%M%S')

                    authors = article.get('authors')
                    summary = article.get('summary')

                    # Initialize a dictionary to hold the topic relevance scores
                    topic_scores = {topic: 0 for topic in topics}

                    # Loop through each topic in the 'topics' list
                    for topic in article.get('topics', []):
                        # If the topic is in the set of topics, update its relevance score
                        if topic['topic'] in topics:
                            topic_scores[topic['topic']] = topic['relevance_score']

                    # Loop through each ticker sentiment in the 'ticker_sentiment' list
                    for ticker_sentiment in article.get('ticker_sentiment', []):
                        # Extract the ticker and sentiment fields
                        ticker_sentiment_ticker = ticker_sentiment.get('ticker')

                        # Skip processing if the ticker sentiment is not for 'AAPL'
                        if ticker_sentiment_ticker != 'AAPL':
                            continue

                        relevance_score = ticker_sentiment.get('relevance_score')
                        ticker_sentiment_score = ticker_sentiment.get('ticker_sentiment_score')
                        ticker_sentiment_label = ticker_sentiment.get('ticker_sentiment_label')

                        # Rest of the code...
                        data.append([ticker, title, url, time_published, authors, summary, relevance_score, ticker_sentiment_score, ticker_sentiment_label] + list(topic_scores.values()))
        except FileNotFoundError:
            print(f"No news file found for ticker: {ticker}")

    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=['Ticker', 'Title', 'URL', 'Time Published', 'Authors', 'Summary', 'Relevance Score', 'Ticker Sentiment Score', 'Ticker Sentiment Label'] + list(topics))

    return df