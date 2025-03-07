from dotenv import load_dotenv
load_dotenv()


examples = [
  {
    "available_tags": ["blog_post", "news_article", "tutorial", "video", "programming", "podcast", "forum_discussion"],
    "question": "How can beginners effectively learn JavaScript?"
  },
  {
    "available_tags": ["research_paper", "opinion_piece", "fiction", "history", "biography", "documentary", "magazine_feature"],
    "question": "What are the underlying factors behind the rise and fall of empires?"
  },
  {
    "available_tags": ["journal_article", "magazine_feature", "novel", "technology", "innovation", "whitepaper", "blog_post"],
    "question": "Which breakthroughs are transforming artificial intelligence?"
  },
  {
    "available_tags": ["scientific_paper", "blog_post", "documentary", "biology", "genetics", "news_article", "research_report"],
    "question": "How do genetic mutations contribute to evolutionary change?"
  },
  {
    "available_tags": ["research_paper", "newspaper_article", "fiction", "economics", "finance", "whitepaper", "case_study"],
    "question": "What are the main economic forces driving inflation today?"
  },
  {
    "available_tags": ["whitepaper", "press_release", "novel", "marketing", "strategy", "blog_post", "podcast"],
    "question": "How do companies build strong brand identities?"
  },
  {
    "available_tags": ["scientific_paper", "news_report", "novel", "astronomy", "astrophysics", "space_exploration", "conference_paper"],
    "question": "What are the latest findings on exoplanets?"
  },
  {
    "available_tags": ["academic_article", "editorial", "novel", "psychology", "sociology", "case_study", "forum_post"],
    "question": "What drives the formation of social groups and community behaviors?"
  },
  {
    "available_tags": ["research_paper", "feature_story", "novel", "environment", "climate", "policy_brief", "news_article"],
    "question": "How are recent policy changes impacting climate action?"
  }
]



CONSTRUCTION_PROMPT = """You are given a list of available tags and a natural language question.
Your task is to generate a filter expression that captures the intent of the question using the provided tags.

available_tags: {available_tags}
question: {question}

The filter expression must follow these rules:
1. **Tag Names:**  
   - Only use tags from the list of available tags provided.

2. **Operators:**  
   - Use the **AND** operator (`&`) to require that multiple tags must be present.  
   - Use the **OR** operator (`|`) to indicate that at least one of several tags can be present.  
   - Use the **NOT** operator (`!`) to exclude a tag from the results.

3. **Grouping:**  
   - Parentheses `(` and `)` may be used to group expressions and control the order of evaluation.

4. **Interpretation:**  
   - Analyze the intent of the question and select tags that best capture that intent.
   - Exclude tags that are not relevant to the question by using the NOT operator.
   - The expression should be as concise as possible while still being semantically correct.

**Example:**

- **Available Tags:** `scientific_paper`, `new_article`, `novel`, `physics`, `chemistry`, `conference_paper`
- **Question:** What causes gravitational lensing in space?
- **Expected Output:** `scientific_paper & physics & !novel`

Using the rules above, generate a reasonable filter expression for any given list of available tags and question.
Be careful to not create a too strict expression that would exclude relevant content.
Your answer should be a valid filter expression. No additonal explanations!"""

from openai import OpenAI

client = OpenAI()

for example in examples:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": CONSTRUCTION_PROMPT.format(**example)}
        ]
    )
    expression = response.choices[0].message.content.strip()
    print(f"Available Tags: {example['available_tags']}")
    print(f"Question: {example['question']}")
    print(f"Expression: {expression}\n")