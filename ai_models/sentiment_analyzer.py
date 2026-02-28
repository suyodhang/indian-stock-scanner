"""
News & Social Media Sentiment Analysis for Indian Stocks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from news and social media"""
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key
        self.sentiment_model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained sentiment model"""
        try:
            from transformers import pipeline
            
            # Use FinBERT for financial sentiment
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None
            )
            logger.info("âœ… FinBERT model loaded for sentiment analysis")
        except ImportError:
            logger.warning("transformers not installed. Using basic sentiment analysis.")
            self.sentiment_model = None
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}. Using basic analysis.")
            self.sentiment_model = None
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if self.sentiment_model:
            result = self.sentiment_model(text[:512])  # FinBERT max length
            
            # FinBERT returns: positive, negative, neutral
            scores = {item['label']: item['score'] for item in result[0]}
            
            return {
                'text': text[:200],
                'positive': scores.get('positive', 0),
                'negative': scores.get('negative', 0),
                'neutral': scores.get('neutral', 0),
                'sentiment': max(scores, key=scores.get),
                'confidence': max(scores.values())
            }
        else:
            return self._basic_sentiment(text)
    
    def _basic_sentiment(self, text: str) -> Dict:
        """Basic rule-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 
            'upgrade', 'outperform', 'buy', 'strong', 'positive',
            'record high', 'breakout', 'momentum', 'upbeat', 'boom',
            'dividend', 'bonus', 'robust', 'expansion'
        ]
        
        negative_words = [
            'bearish', 'crash', 'fall', 'loss', 'decline', 'downgrade',
            'sell', 'weak', 'negative', 'slump', 'correction', 'risk',
            'debt', 'default', 'scam', 'fraud', 'penalty', 'ban',
            'recession', 'inflation', 'crisis'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total = pos_count + neg_count + 1
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = 1 - pos_score - neg_score
        
        if pos_score > neg_score:
            sentiment = 'positive'
        elif neg_score > pos_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'text': text[:200],
            'positive': pos_score,
            'negative': neg_score,
            'neutral': neu_score,
            'sentiment': sentiment,
            'confidence': max(pos_score, neg_score, neu_score)
        }
    
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch news articles for a stock"""
        articles = []
        
        # Method 1: NewsAPI
        if self.news_api_key:
            try:
                from newsapi import NewsApiClient
                newsapi = NewsApiClient(api_key=self.news_api_key)
                
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                response = newsapi.get_everything(
                    q=f"{symbol} stock NSE BSE India",
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                for article in response.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                    })
            except Exception as e:
                logger.error(f"Error fetching news from NewsAPI: {e}")
        
        # Method 2: Google News RSS (free, no API key needed)
        try:
            import feedparser
            
            query = f"{symbol}+stock+india"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:15]:
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'source': entry.get('source', {}).get('title', 'Google News'),
                    'url': entry.get('link', ''),
                    'published_at': entry.get('published', ''),
                })
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
        
        return articles
    
    def get_stock_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Get overall sentiment for a stock
        
        Returns:
            Aggregated sentiment scores and individual article sentiments
        """
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'article_count': 0,
                'articles': []
            }
        
        sentiments = []
        for article in articles:
            text = f"{article['title']}. {article.get('description', '')}"
            sentiment = self.analyze_text(text)
            article['sentiment'] = sentiment
            sentiments.append(sentiment)
        
        # Aggregate sentiments
        avg_positive = np.mean([s['positive'] for s in sentiments])
        avg_negative = np.mean([s['negative'] for s in sentiments])
        avg_neutral = np.mean([s['neutral'] for s in sentiments])
        
        # Composite score: -1 (very negative) to +1 (very positive)
        sentiment_score = avg_positive - avg_negative
        
        if sentiment_score > 0.1:
            overall = 'positive'
        elif sentiment_score < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall,
            'sentiment_score': float(sentiment_score),
            'avg_positive': float(avg_positive),
            'avg_negative': float(avg_negative),
            'avg_neutral': float(avg_neutral),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'article_count': len(articles),
            'articles': articles
        }
    
    def batch_sentiment_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple stocks"""
        results = []
        
        for symbol in symbols:
            logger.info(f"Analyzing sentiment for {symbol}...")
            sentiment = self.get_stock_sentiment(symbol)
            results.append({
                'symbol': symbol,
                'sentiment': sentiment['overall_sentiment'],
                'score': sentiment['sentiment_score'],
                'positive': sentiment['avg_positive'],
                'negative': sentiment['avg_negative'],
                'articles': sentiment['article_count']
            })
        
        return pd.DataFrame(results)
