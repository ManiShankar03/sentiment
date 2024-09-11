import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging
from datetime import datetime, timedelta
from typing import Tuple, List
import numpy as np
from scipy import stats


# Set page configuration
st.set_page_config(layout="wide", page_title="Financial News Sentiment Analysis")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Define colors
BLUE = '#1E90FF'
GREEN = '#00FF00'
RED = '#FF0000'
LIGHT_GREY = '#A9A9A9'
BLACK = '#000000'
WHITE = '#FFFFFF'

# Custom CSS to set the background color and style news cards
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: white;
}
.news-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}
.news-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}
.news-sentiment {
    font-size: 14px;
    margin-bottom: 10px;
}
.news-link {
    color: #1E90FF;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# Load the FinBERT model
@st.cache_resource
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the sentiment analysis model. Please try again later.")
        return None

# Load and process data
@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('financial_news_data.csv')
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        # Add a 'company_name' column (you'll need to provide this mapping)
        df['company_name'] = df['ticker'].map({
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'VZ': 'Verizon Communications Inc.',
            'HD': 'Home Depot Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc.',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'GE': 'General Electric Co.',
            'XOM': 'Exxon Mobil Corp.'
        })
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Failed to load the data. Please check if the file exists and try again.")
        return pd.DataFrame()

# Function to get improved sentiment
def get_improved_sentiment(text: str, pipe, threshold_positive: float = 0.1, threshold_negative: float = 0.1) -> Tuple[str, float]:
    result = pipe(text)[0]
    sentiment_label = result['label']
    sentiment_score = result['score']

    if sentiment_label == 'positive':
        adjusted_score = sentiment_score
    elif sentiment_label == 'negative':
        adjusted_score = -sentiment_score
    else:
        adjusted_score = (sentiment_score - 0.5) * 0.4

    if adjusted_score > threshold_positive:
        sentiment_label = 'positive'
    elif adjusted_score < -threshold_negative:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'

    return sentiment_label, adjusted_score

# Function to create sentiment score chart
def create_sentiment_score_chart(company_data: pd.DataFrame, selected_company: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=company_data['published_utc'],
        y=company_data['finbert_sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color=BLUE, width=2),
        marker=dict(color=BLUE, size=6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Score:</b> %{y:.2f}<br><b>Title:</b> %{text}<extra></extra>',
        text=company_data['title']
    ))

    fig.update_layout(
        title=f"Sentiment Score Trend for {selected_company}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        plot_bgcolor=BLACK,
        paper_bgcolor=BLACK,
        font=dict(family="Arial", size=12, color=WHITE),
        hovermode='closest',
        yaxis=dict(range=[-1, 1], gridcolor=LIGHT_GREY, zerolinecolor=LIGHT_GREY),
        xaxis=dict(gridcolor=LIGHT_GREY, zerolinecolor=LIGHT_GREY),
        title_font=dict(size=24, color=WHITE, family="Arial"),
        xaxis_title_font=dict(size=14, color=WHITE),
        yaxis_title_font=dict(size=14, color=WHITE),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.add_hline(y=0, line_dash="dash", line_color=LIGHT_GREY, opacity=0.5)

    return fig

# Function to create rolling average sentiment trend chart
def create_rolling_avg_chart(company_data: pd.DataFrame, selected_company: str, window_size: int) -> go.Figure:
    company_data['rolling_avg'] = company_data['finbert_sentiment_score'].rolling(window=window_size).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=company_data['published_utc'],
        y=company_data['rolling_avg'],
        mode='lines',
        name='Rolling Average',
        line=dict(color=GREEN, width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Rolling Avg Score:</b> %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{window_size}-Day Rolling Average Sentiment Trend for {selected_company}",
        xaxis_title="Date",
        yaxis_title=f"{window_size}-Day Rolling Average Sentiment Score",
        plot_bgcolor=BLACK,
        paper_bgcolor=BLACK,
        font=dict(family="Arial", size=12, color=WHITE),
        hovermode='closest',
        yaxis=dict(range=[-1, 1], gridcolor=LIGHT_GREY, zerolinecolor=LIGHT_GREY),
        xaxis=dict(gridcolor=LIGHT_GREY, zerolinecolor=LIGHT_GREY),
        title_font=dict(size=24, color=WHITE, family="Arial"),
        xaxis_title_font=dict(size=14, color=WHITE),
        yaxis_title_font=dict(size=14, color=WHITE),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.add_hline(y=0, line_dash="dash", line_color=LIGHT_GREY, opacity=0.5)

    return fig

# Function to create sentiment distribution donut chart
def create_sentiment_distribution_chart(company_data: pd.DataFrame, selected_company: str) -> go.Figure:
    sentiment_counts = company_data['finbert_sentiment'].value_counts()
    colors = [GREEN, BLUE, RED]
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.5,
        marker_colors=colors,
        textinfo='percent+label',
        textfont=dict(size=14, color=BLACK),
        hoverinfo='label+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"Sentiment Distribution for {selected_company}",
        plot_bgcolor=BLACK,
        paper_bgcolor=BLACK,
        font=dict(family="Arial", size=12, color=WHITE),
        title_font=dict(size=24, color=WHITE, family="Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=WHITE)),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Function to display news articles alternating between positive and negative
def display_news_articles(company_data: pd.DataFrame, num_articles: int = 10):
    st.subheader("Top News Articles")
    
    # Sort articles by absolute sentiment score
    sorted_articles = company_data.sort_values(by='finbert_sentiment_score', key=abs, ascending=False)
    
    positive_articles = sorted_articles[sorted_articles['finbert_sentiment'] == 'positive']
    negative_articles = sorted_articles[sorted_articles['finbert_sentiment'] == 'negative']
    
    displayed_count = 0
    positive_index = 0
    negative_index = 0
    
    while displayed_count < num_articles and (positive_index < len(positive_articles) or negative_index < len(negative_articles)):
        if displayed_count % 2 == 0 and positive_index < len(positive_articles):
            article = positive_articles.iloc[positive_index]
            positive_index += 1
        elif negative_index < len(negative_articles):
            article = negative_articles.iloc[negative_index]
            negative_index += 1
        else:
            article = positive_articles.iloc[positive_index]
            positive_index += 1
        
        sentiment_color = GREEN if article['finbert_sentiment'] == 'positive' else RED
        
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{article['title']}</div>
            <div class="news-sentiment" style="color: {sentiment_color};">
                Sentiment: {article['finbert_sentiment'].capitalize()} (Score: {article['finbert_sentiment_score']:.2f})
            </div>
            <a href="{article['article_url']}" target="_blank" class="news-link">Read full article</a>
        </div>
        """, unsafe_allow_html=True)
        
        displayed_count += 1

# Function to generate chart conclusion
def generate_chart_conclusion(chart_data: pd.DataFrame, chart_type: str) -> str:
    if chart_type == "sentiment_score":
        recent_sentiment = chart_data['finbert_sentiment_score'].iloc[-1]
        avg_sentiment = chart_data['finbert_sentiment_score'].mean()
        
        if recent_sentiment > avg_sentiment:
            return "The recent sentiment is more positive than the average, indicating an improving trend."
        elif recent_sentiment < avg_sentiment:
            return "The recent sentiment is more negative than the average, suggesting a declining trend."
        else:
            return "The recent sentiment is in line with the average, indicating a stable trend."
    
    elif chart_type == "rolling_average":
        recent_avg = chart_data['rolling_avg'].iloc[-1]
        
        if recent_avg > 0.1:
            return "The recent rolling average sentiment is positive, suggesting overall optimistic market perception."
        elif recent_avg < -0.1:
            return "The recent rolling average sentiment is negative, indicating overall pessimistic market perception."
        else:
            return "The recent rolling average sentiment is neutral, suggesting balanced market perception."
    
    elif chart_type == "sentiment_distribution":
        sentiment_counts = chart_data['finbert_sentiment'].value_counts()
        total_articles = len(chart_data)
        
        positive_percentage = (sentiment_counts.get('positive', 0) / total_articles) * 100
        negative_percentage = (sentiment_counts.get('negative', 0) / total_articles) * 100
        
        if positive_percentage > negative_percentage:
            return f"The sentiment is predominantly positive ({positive_percentage:.1f}% of articles), indicating favorable market perception."
        elif negative_percentage > positive_percentage:
            return f"The sentiment is predominantly negative ({negative_percentage:.1f}% of articles), suggesting challenges in market perception."
        else:
            return "The sentiment is evenly distributed between positive and negative, indicating mixed market perception."

# Function to calculate sentiment trend
def calculate_sentiment_trend(company_data: pd.DataFrame) -> str:
    sentiment_scores = company_data['finbert_sentiment_score'].values
    x = np.arange(len(sentiment_scores))
    slope, _, _, _, _ = stats.linregress(x, sentiment_scores)
    
    if slope > 0.01:
        return "improving"
    elif slope < -0.01:
        return "declining"
    else:
        return "stable"

# Function to generate additional insights
def generate_additional_insights(company_data: pd.DataFrame) -> str:
    sentiment_trend = calculate_sentiment_trend(company_data)
    recent_sentiment = company_data['finbert_sentiment_score'].iloc[-5:].mean()
    volatility = company_data['finbert_sentiment_score'].std()
    
    insights = f"Additional Insights:\n\n"
    insights += f"1. Sentiment Trend: The overall sentiment trend appears to be {sentiment_trend}.\n"
    
    if recent_sentiment > 0.1:
        insights += "2. Recent Sentiment: The most recent articles show a positive bias, which could indicate improving perception.\n"
    elif recent_sentiment < -0.1:
        insights += "2. Recent Sentiment: The most recent articles show a negative bias, which might suggest emerging concerns.\n"
    else:
        insights += "2. Recent Sentiment: The most recent articles show a neutral stance, indicating balanced recent coverage.\n"
    
    if volatility > 0.5:
        insights += "3. Sentiment Volatility: High volatility in sentiment scores suggests rapidly changing perceptions or conflicting news.\n"
    else:
        insights += "3. Sentiment Volatility: Low volatility in sentiment scores indicates relatively consistent market perception.\n"
    
    return insights

# Function to generate a more balanced final conclusion (continued)
def generate_final_conclusion(company_data: pd.DataFrame, selected_company: str) -> str:
    sentiment_counts = company_data['finbert_sentiment'].value_counts()
    total_articles = len(company_data)
    
    positive_percentage = (sentiment_counts.get('positive', 0) / total_articles) * 100
    negative_percentage = (sentiment_counts.get('negative', 0) / total_articles) * 100
    neutral_percentage = (sentiment_counts.get('neutral', 0) / total_articles) * 100
    
    avg_sentiment_score = company_data['finbert_sentiment_score'].mean()
    
    conclusion = f"Final Conclusion for {selected_company}:\n\n"
    
    # Adjust for potential positive bias
    adjusted_positive_percentage = max(positive_percentage - 15, 0)  # Reduce positive percentage by 15%
    adjusted_negative_percentage = min(negative_percentage + 5, 100)  # Increase negative percentage by 5%
    
    if adjusted_positive_percentage > adjusted_negative_percentage + 20:
        sentiment_trend = "predominantly positive"
    elif adjusted_negative_percentage > adjusted_positive_percentage + 10:
        sentiment_trend = "leaning negative"
    else:
        sentiment_trend = "relatively balanced"
    
    conclusion += f"The overall sentiment for {selected_company} appears to be {sentiment_trend}. "
    conclusion += f"After adjusting for potential model bias, approximately {adjusted_positive_percentage:.1f}% of articles show positive sentiment, "
    conclusion += f"while {adjusted_negative_percentage:.1f}% indicate negative sentiment.\n\n"
    
    if avg_sentiment_score > 0.1:
        conclusion += "The average sentiment score is slightly positive, suggesting cautious optimism in the market. "
    elif avg_sentiment_score < -0.1:
        conclusion += "The average sentiment score is slightly negative, indicating some market concerns. "
    else:
        conclusion += "The average sentiment score is close to neutral, reflecting a mix of positive and negative factors. "
    
    
    return conclusion

# Main Streamlit app
def main():
    st.title("Financial News Sentiment Analysis Dashboard")

    # Load data and model
    df = load_data()
    pipe = load_model()

    if df.empty or pipe is None:
        return

    # Get the list of company names
    companies = df['company_name'].unique().tolist()

    # Sidebar for company selection and date range
    selected_company = st.sidebar.selectbox("Select a company", companies)
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(df['published_utc'].min().date(), df['published_utc'].max().date()),
        min_value=df['published_utc'].min().date(),
        max_value=df['published_utc'].max().date()
    )

    # Filter data for selected company and date range
    start_date, end_date = date_range
    company_data = df[(df['company_name'] == selected_company) & 
                      (df['published_utc'].dt.date >= start_date) & 
                      (df['published_utc'].dt.date <= end_date)].sort_values('published_utc')

    # Apply sentiment analysis if not already present
    if 'finbert_sentiment' not in company_data.columns or 'finbert_sentiment_score' not in company_data.columns:
        company_data['finbert_sentiment'], company_data['finbert_sentiment_score'] = zip(*company_data['title'].apply(lambda x: get_improved_sentiment(x, pipe)))

    # Display charts
    sentiment_score_chart = create_sentiment_score_chart(company_data, selected_company)
    st.plotly_chart(sentiment_score_chart, use_container_width=True)
    st.markdown(generate_chart_conclusion(company_data, "sentiment_score"))
    
    window_size = min(30, len(company_data))
    rolling_avg_chart = create_rolling_avg_chart(company_data, selected_company, window_size)
    st.plotly_chart(rolling_avg_chart, use_container_width=True)
    st.markdown(generate_chart_conclusion(company_data, "rolling_average"))
    
    sentiment_distribution_chart = create_sentiment_distribution_chart(company_data, selected_company)
    st.plotly_chart(sentiment_distribution_chart, use_container_width=True)
    st.markdown(generate_chart_conclusion(company_data, "sentiment_distribution"))

    # Display sentiment counts
    st.subheader(f"Sentiment Counts for {selected_company}")
    sentiment_counts = company_data['finbert_sentiment'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", sentiment_counts.get('positive', 0), delta=f"{(sentiment_counts.get('positive', 0) / len(company_data) * 100):.1f}%")
    col2.metric("Neutral", sentiment_counts.get('neutral', 0), delta=f"{(sentiment_counts.get('neutral', 0) / len(company_data) * 100):.1f}%")
    col3.metric("Negative", sentiment_counts.get('negative', 0), delta=f"{(sentiment_counts.get('negative', 0) / len(company_data) * 100):.1f}%")

    # Display news articles
    display_news_articles(company_data)

    # Display performance insights
    st.subheader("Performance Insights")
    total_articles = len(company_data)
    positive_percentage = (sentiment_counts.get('positive', 0) / total_articles) * 100
    neutral_percentage = (sentiment_counts.get('neutral', 0) / total_articles) * 100
    negative_percentage = (sentiment_counts.get('negative', 0) / total_articles) * 100

    insight_text = f"""
    During the selected period from {start_date} to {end_date}, the sentiment analysis for {selected_company} shows:
    
    - Positive sentiment: {positive_percentage:.1f}% of articles
    - Neutral sentiment: {neutral_percentage:.1f}% of articles
    - Negative sentiment: {negative_percentage:.1f}% of articles
    
    Total articles analyzed: {total_articles}
    """

    if positive_percentage > 60:
        st.success(insight_text)
    elif negative_percentage > 50:
        st.error(insight_text)
    else:
        st.info(insight_text)

    # Display additional insights
    st.subheader("Additional Insights")
    st.markdown(generate_additional_insights(company_data))

    # Display final conclusion
    st.subheader("Final Conclusion")
    st.markdown(generate_final_conclusion(company_data, selected_company))

    # Add a footer
    st.markdown("---")
    st.markdown("© 2023 Financial News Sentiment Analysis Dashboard. All rights reserved.")

if __name__ == '__main__':
    main()