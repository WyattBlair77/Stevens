import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, regexp_replace, split, mean, udf
from pyspark.sql.types import IntegerType
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import SentenceDetector, Tokenizer, Lemmatizer, SentimentDetector
from pyspark.ml import Pipeline

# Initialize Spark
spark = SparkSession.builder \
    .appName("GutenbergSentimentAnalysisWithSparkNLP") \
    .master("local[*]") \
    .getOrCreate()

# Read multiple parts of gutenberg data
df = spark.read.csv("gutenberg_data_part*.csv", header=True, inferSchema=True)
df = df.withColumn("Year", col("Year").cast(IntegerType()))

# Basic text cleaning: remove punctuation and lowercase
# We assume 'Text' is the column with the book text
df_clean = df.withColumn("clean_text", lower(regexp_replace("Text", "[^a-zA-Z0-9\\s]", " ")))

# Set up the Spark NLP pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("clean_text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Provide a lemma dictionary file accessible to the cluster
lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("lemmas_small.txt", key_delimiter="->", value_delimiter="\t")

# Provide a sentiment dictionary file accessible to the cluster
sentiment_detector = SentimentDetector() \
    .setInputCols(["lemma", "sentence"]) \
    .setOutputCol("sentiment_score") \
    .setDictionary("default-sentiment-dict.txt", ",")

finisher = Finisher() \
    .setInputCols(["sentiment_score"]) \
    .setOutputCols(["sentiment"])

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    lemmatizer,
    sentiment_detector,
    finisher
])

# Fit the pipeline (if it's using transformers that need fitting)
# In this case, no training is typically required for these annotators,
# so we can fit on a small subset or the same df.
model = pipeline.fit(df_clean)

# Transform the DataFrame to get sentiment column
df_sent = model.transform(df_clean)

# df_sent now has a 'sentiment' column (an array) - let's get the first value
# SentimentDetector with a dictionary can produce categories like "positive", "negative"
# but it depends on the dictionary provided. Often it's a score. 
# Check what the dictionary produces and adjust accordingly.
# Usually, Finisher flattens arrays, so 'sentiment' might be a single-value array.
df_sent = df_sent.withColumn("sentiment", df_sent["sentiment"].getItem(0))

# Now we have a column 'sentiment' which should have sentiment results (e.g., positive/negative or a score)

# Topics we are interested in
topics = ["gender", "politics", "education", "technology", "religion", "medicine", "science", "philosophy"]

# Explode tokens again (we have 'token' from the pipeline)
# Note: After the pipeline, we have token annotations. Finisher was only applied to sentiment score.
# The 'token' column from the pipeline is an annotation column, we need to extract text from it.
# By default, Tokenizer produces annotations. To get the actual token text, you can use the 
# 'finished_tokens' approach by adding another Finisher stage or just re-tokenize using split:
# Since we already used a pipeline, let's do a simple split on `clean_text` for topic filtering.
df_exploded = df_sent.withColumn("tok", explode(split(col("clean_text"), "\\s+")))

# Filter for topics
df_topics = df_exploded.filter(lower(col("tok")).isin([t.lower() for t in topics]))

# Group by year and token to find average sentiment.
# If 'sentiment' is a categorical label (positive/negative), we need a strategy:
# - If sentiment is numeric score: mean is straightforward.
# - If sentiment is a label: consider mapping "positive" = 1, "negative" = -1, "neutral"=0, etc.
# For simplicity, assume sentiment is a numeric score from the dictionary. 
# If it's not, you'll need to map them first.

# If sentiment is a string like "positive", "negative", we must convert it:
sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
map_sentiment_udf = udf(lambda s: sentiment_map.get(s, 0.0))
df_topics_scored = df_topics.withColumn("sentiment_score", map_sentiment_udf(col("sentiment")))

df_topic_sentiment = df_topics_scored.groupBy("Year", "tok").agg(mean("sentiment_score").alias("avg_sentiment"))

# Convert to Pandas for plotting
topic_pdf = df_topic_sentiment.toPandas()

# Normalize token casing
topic_pdf["tok"] = topic_pdf["tok"].str.lower()

# Pivot for easier plotting
pivot_df = topic_pdf.pivot(index="Year", columns="tok", values="avg_sentiment")
pivot_df = pivot_df.sort_values(by="Year")

# Plot the results
plt.figure(figsize=(10, 6))
for topic in topics:
    if topic in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[topic], label=topic)

plt.title("Average Sentiment Over Time For Selected Topics (Spark NLP)")
plt.xlabel("Year")
plt.ylabel("Average Sentiment Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("topic_sentiment_over_time_sparknlp.png")
plt.show()

spark.stop()
