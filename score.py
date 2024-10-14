# Import necessary libraries
from llama_cpp import Llama, LlamaGrammar
import sqlite3
import csv
import os

# CSV file to store the scores
csv_file_path = '/root/senti-main/news_sentiment_scores.csv'

# Connect to the existing SQLite database
conn = sqlite3.connect('/root/senti-main/reuters.db')
cursor = conn.cursor()

# Fetch all news articles starting from news_id 6000
cursor.execute("SELECT news_id, headline, text FROM news WHERE news_id >= 601 AND news_id <= 3000 ORDER BY news_id")
news_articles = cursor.fetchall()

# Create or connect to the new database where the sentiment will be stored
conn_sentiment = sqlite3.connect('/root/senti-main/news_sentiment.db')
cursor_sentiment = conn_sentiment.cursor()

# Create a table to store the news and its sentiment score (if not already exists)
cursor_sentiment.execute('''
    CREATE TABLE IF NOT EXISTS news_sentiment (
        news_id INTEGER PRIMARY KEY,
        sentiment REAL
    )
''')

# Open the CSV file to store the sentiment scores in append mode
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # If the file is newly created (no header), add the header
    csv_writer.writerow(['news_id', 'headline', 'sentiment'])
    
    # Prompt template
    prompt_template = """
    News Article:
    Headline: {headline}

    Text: {text}

    Based on the above news article, provide a yes/no answer for each of these points.

    Topic: Financial Market Sentiment Direction
    1. Does the news indicate a significant increase in corporate earnings or profits for major companies?
    2. Does the news announce new government stimulus or economic support measures (e.g., tax cuts, subsidies)?
    3. Is the news about a breakthrough in trade negotiations or the signing of a major trade agreement?
    4. Does the news report significant technological advancements or product launches from major tech companies?
    5. Does the news highlight a major decrease in unemployment rates or a significant rise in employment figures?
    6. Does the news reveal a substantial drop in corporate earnings or profits for major companies?
    7. Is the news about the imposition of new trade barriers or tariffs?
    8. Does the news indicate a significant increase in interest rates by major central banks?
    9. Is the news about escalating geopolitical tensions or the outbreak of conflict?
    10. Does the news report a significant rise in inflation rates or expectations?

    Respond in this way:
        - Provide your reasoning for each question.
        - THEN provide a final answer for the yes/no question.
    """

    root_rule = 'root ::= ' + ' '.join([f'"Reasoning {i} (1 sentence max): " reasoning "\\n" "Answer {i} (yes/no): " yesno "\\n"' for i in range(1, 11)])

    grammar = f"""
    {root_rule}
    reasoning ::= sentence
    sentence ::= [^.!?]+[.!?]
    yesno ::= "yes" | "no"
    """

    grammar = LlamaGrammar.from_string(grammar)

    # Load the Llama model
    llm = Llama(
        model_path = r"/root/senti-main/Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf",
        n_gpu_layers = -1,
        seed = 123,
        n_ctx = 8192,
        verbose = False
    )

    # Process each article and calculate sentiment
    for news_id, headline, text in news_articles:
        try:
            prompt = prompt_template.format(headline=headline, text=text)

            output = llm(
                prompt,
                max_tokens=2000,
                stop=["</s>"],
                echo=False,
                grammar=grammar
            )

            response = output["choices"][0]["text"].strip()
            yes_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "yes")
            no_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "no")

            final_score = 0
            for i, line in enumerate(response.split('\n')):
                if line.startswith("Answer"):
                    answer = line.split(": ")[1].strip().lower()
                    if answer == "yes":
                        if 1 <= (i // 2) + 1 <= 5:  # Questions 1 to 5: A yes here gets +1 for positive sentiment
                            final_score += 1
                        elif 6 <= (i // 2) + 1 <= 10:  # Questions 6 to 10: A yes here gets -1 for negative sentiment
                            final_score -= 1

            # Store the final score in the new database
            cursor_sentiment.execute('''
                INSERT OR REPLACE INTO news_sentiment (news_id, sentiment)
                VALUES (?, ?)
            ''', (news_id, final_score))

            # Store the result in CSV
            csv_writer.writerow([news_id, headline, final_score])

        except Exception as e:
            # If an error occurs, store NULL for the sentiment and continue
            print(f"Error processing record {news_id}: {e}")
            cursor_sentiment.execute('''
                INSERT OR REPLACE INTO news_sentiment (news_id, sentiment)
                VALUES (?, ?)
            ''', (news_id, None))

            # Store NULL result in CSV
            csv_writer.writerow([news_id, headline, None])

        # Print simple message
        print(f"Record {news_id} processed and stored.")

# Commit the changes and close the database connections
conn_sentiment.commit()
conn_sentiment.close()
conn.close()

print(f"Sentiment scores have been stored in '{csv_file_path}'.")
