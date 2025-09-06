from langchain.tools import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import polars as pl
from datetime import datetime

# Import the full email DataFrame
from lib.load_data import df

# Import helper functions
from lib.utils import normalize_list, match_value_in_columns

# Initialize the VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(score):
    """Classifies a VADER compound score into a category."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@tool
def sentiment_analysis_tool(
    sender: str = None,
    recipient: str = None,
    start_date: str = None,
    end_date: str = None,
    timeline_granularity: str = None
) -> str:
    """
    Analyzes the sentiment of emails based on filters like sender, recipient, or date range.
    It can provide an overall summary or a timeline of sentiment.
    Use this for high-level questions like "what is the overall sentiment in customer emails?" or "show me the sentiment timeline for emails from Raja."

    Args:
        sender (str, optional): Filter emails by the sender's name or email.
        recipient (str, optional): Filter emails by the recipient's name or email.
        start_date (str, optional): The start date for the analysis (YYYY-MM-DD).
        end_date (str, optional): The end date for the analysis (YYYY-MM-DD).
        timeline_granularity (str, optional): If provided, groups sentiment by 'month' or 'year'. If None, provides an overall summary.
    """
    print(f"sentiment_analysis_tool called with: sender={sender}, start_date={start_date}, end_date={end_date}, timeline={timeline_granularity}")
    
    if df.is_empty():
        return "Error: Email data is not loaded."

    temp_df = df.clone()
    mask = pl.lit(True)

    # --- 1. Filter the DataFrame (Unchanged) ---
    if sender:
        sender_lower = sender.lower()
        temp_df = temp_df.with_columns(pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized"))
        mask = mask & pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender_lower, x), return_dtype=bool)
    
    if start_date or end_date:
        temp_df = temp_df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False).alias("date_dt"))
    if start_date:
        mask = mask & (pl.col("date_dt") >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        mask = mask & (pl.col("date_dt") <= datetime.strptime(end_date, "%Y-%m-%d"))

    filtered_df = temp_df.filter(mask)

    if filtered_df.is_empty():
        return "No emails found for the specified criteria to analyze sentiment."

    # --- 2. Map & Analyze: Run VADER on each email body (THE FIX IS HERE) ---
    
    # --- THE FIX: Safely extract the text body ---
    # This new code checks if the 'body' column is a struct and not null.
    # If it is, it extracts the 'text' field. Otherwise, it defaults to an empty string.
    safe_text_extraction_expr = (
        pl.when(pl.col("body").is_not_null() & (pl.col("body").dtype == pl.Struct))
        .then(pl.col("body").struct.field("text"))
        .otherwise(pl.lit(""))
    )

    sentiments = filtered_df.select(
        pl.col("date_dt").alias("date"),
        safe_text_extraction_expr.map_elements(
            lambda text: analyzer.polarity_scores(text or "")['compound'], # Ensure text is not None
            return_dtype=pl.Float64
        ).alias("sentiment_score")
    ).drop_nulls()

    if sentiments.is_empty():
        return "Found emails, but could not extract text bodies to analyze sentiment."

    # --- 3. Reduce & Synthesize the results (Unchanged) ---
    if timeline_granularity in ["month", "year"]:
        period = "1mo" if timeline_granularity == "month" else "1y"
        timeline_summary = sentiments.group_by_dynamic("date", every=period).agg(
            pl.mean("sentiment_score").alias("average_sentiment"),
            pl.count().alias("email_count")
        ).sort("date")
        
        summary_lines = [f"Sentiment Timeline Analysis (granularity: {timeline_granularity}):"]
        for row in timeline_summary.to_dicts():
            period_str = row['date'].strftime('%Y-%m' if timeline_granularity == 'month' else '%Y')
            avg_sentiment = row['average_sentiment']
            sentiment_class = classify_sentiment(avg_sentiment)
            summary_lines.append(
                f"- Period: {period_str}, Email Count: {row['email_count']}, "
                f"Average Sentiment: {avg_sentiment:.2f} ({sentiment_class})"
            )
        return "\n".join(summary_lines)

    else:
        overall_summary = sentiments.select(
            pl.mean("sentiment_score").alias("average_sentiment"),
            pl.col("sentiment_score").map_elements(lambda s: classify_sentiment(s), return_dtype=str).value_counts().alias("sentiment_counts"),
            pl.count().alias("total_emails")
        ).to_dicts()[0]

        avg_score = overall_summary['average_sentiment']
        total_emails = overall_summary['total_emails']
        counts = {d['sentiment_score']: d['count'] for d in overall_summary['sentiment_counts']}
        
        summary = (
            f"Overall Sentiment Analysis Summary:\n"
            f"- Total Emails Analyzed: {total_emails}\n"
            f"- Average Sentiment Score: {avg_score:.2f} ({classify_sentiment(avg_score)})\n"
            f"- Positive Emails: {counts.get('Positive', 0)}\n"
            f"- Negative Emails: {counts.get('Negative', 0)}\n"
            f"- Neutral Emails: {counts.get('Neutral', 0)}"
        )
        return summary


# from langchain.tools import tool
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import polars as pl
# from datetime import datetime

# # Import the full email DataFrame from your central data loader
# # We rename it to 'df' to keep the code clean, just like in your other tools.
# from lib.load_data import df

# # Import helper functions from utils
# from lib.utils import normalize_list, match_value_in_columns

# # Initialize the VADER analyzer once
# analyzer = SentimentIntensityAnalyzer()

# def classify_sentiment(score):
#     """Classifies a VADER compound score into a category."""
#     if score >= 0.05:
#         return "Positive"
#     elif score <= -0.05:
#         return "Negative"
#     else:
#         return "Neutral"

# @tool
# def sentiment_analysis_tool(
#     sender: str = None,
#     recipient: str = None,
#     start_date: str = None,
#     end_date: str = None,
#     timeline_granularity: str = None
# ) -> str:
#     """
#     Analyzes the sentiment of emails based on filters like sender, recipient, or date range.
#     It can provide an overall summary or a timeline of sentiment.
#     Use this for high-level questions like "what is the overall sentiment in customer emails?" or "show me the sentiment timeline for emails from Raja."

#     Args:
#         sender (str, optional): Filter emails by the sender's name or email.
#         recipient (str, optional): Filter emails by the recipient's name or email.
#         start_date (str, optional): The start date for the analysis (YYYY-MM-DD).
#         end_date (str, optional): The end date for the analysis (YYYY-MM-DD).
#         timeline_granularity (str, optional): If provided, groups sentiment by 'month' or 'year'. If None, provides an overall summary.
#     """
#     print(f"sentiment_analysis_tool called with: sender={sender}, start_date={start_date}, end_date={end_date}, timeline={timeline_granularity}")
    
#     if df.is_empty():
#         return "Error: Email data is not loaded."

#     temp_df = df.clone()
#     mask = pl.lit(True)

#     # --- 1. Filter the DataFrame based on provided arguments ---
#     # This reuses the same robust filtering logic from your metadata_filtering_tool
#     if sender:
#         sender_lower = sender.lower()
#         temp_df = temp_df.with_columns(pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized"))
#         mask = mask & pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender_lower, x), return_dtype=bool)
    
#     # (You can add recipient filtering here as well if needed)

#     if start_date or end_date:
#         temp_df = temp_df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False).alias("date_dt"))
#     if start_date:
#         mask = mask & (pl.col("date_dt") >= datetime.strptime(start_date, "%Y-%m-%d"))
#     if end_date:
#         mask = mask & (pl.col("date_dt") <= datetime.strptime(end_date, "%Y-%m-%d"))

#     filtered_df = temp_df.filter(mask)

#     if filtered_df.is_empty():
#         return "No emails found for the specified criteria to analyze sentiment."

#     # --- 2. Map & Analyze: Run VADER on each email body ---
#     # We extract the text body and calculate the sentiment score for each email
#     sentiments = filtered_df.select(
#         pl.col("date_dt").alias("date"),
#         pl.col("body").struct.field("text").map_elements(
#             lambda text: analyzer.polarity_scores(text)['compound'],
#             return_dtype=pl.Float64
#         ).alias("sentiment_score")
#     ).drop_nulls()

#     if sentiments.is_empty():
#         return "Found emails, but could not extract text bodies to analyze sentiment."

#     # --- 3. Reduce & Synthesize the results ---
#     if timeline_granularity in ["month", "year"]:
#         # --- Timeline Analysis ---
#         period = "1mo" if timeline_granularity == "month" else "1y"
#         timeline_summary = sentiments.group_by_dynamic("date", every=period).agg(
#             pl.mean("sentiment_score").alias("average_sentiment"),
#             pl.count().alias("email_count")
#         ).sort("date")
        
#         # Format the timeline into a string for the LLM
#         summary_lines = [f"Sentiment Timeline Analysis (granularity: {timeline_granularity}):"]
#         for row in timeline_summary.to_dicts():
#             period_str = row['date'].strftime('%Y-%m' if timeline_granularity == 'month' else '%Y')
#             avg_sentiment = row['average_sentiment']
#             sentiment_class = classify_sentiment(avg_sentiment)
#             summary_lines.append(
#                 f"- Period: {period_str}, Email Count: {row['email_count']}, "
#                 f"Average Sentiment: {avg_sentiment:.2f} ({sentiment_class})"
#             )
#         return "\n".join(summary_lines)

#     else:
#         # --- Overall Summary Analysis ---
#         overall_summary = sentiments.select(
#             pl.mean("sentiment_score").alias("average_sentiment"),
#             pl.col("sentiment_score").map_elements(lambda s: classify_sentiment(s), return_dtype=str).value_counts().alias("sentiment_counts"),
#             pl.count().alias("total_emails")
#         ).to_dicts()[0]

#         avg_score = overall_summary['average_sentiment']
#         total_emails = overall_summary['total_emails']
#         counts = {d['sentiment_score']: d['count'] for d in overall_summary['sentiment_counts']}
        
#         # Format the summary into a string for the LLM
#         summary = (
#             f"Overall Sentiment Analysis Summary:\n"
#             f"- Total Emails Analyzed: {total_emails}\n"
#             f"- Average Sentiment Score: {avg_score:.2f} ({classify_sentiment(avg_score)})\n"
#             f"- Positive Emails: {counts.get('Positive', 0)}\n"
#             f"- Negative Emails: {counts.get('Negative', 0)}\n"
#             f"- Neutral Emails: {counts.get('Neutral', 0)}"
#         )
#         return summary