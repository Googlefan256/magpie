from typing import List, Literal
from pydantic import BaseModel


class DifficultyRating(BaseModel):
    intent: str
    knowledge: str
    difficluty: Literal[
        "very easy",
        "easy",
        "medium",
        "hard",
        "very hard",
    ]


def input_difficulty_rating(inputs: str):
    return (
        f"""
# Instruction

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{inputs}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
""",
        DifficultyRating,
    )


type Tag = Literal[
    "Information seeking",
    "Reasoning",
    "Planning",
    "Editing",
    "Coding & Debugging",
    "Math",
    "Role playing",
    "Data analysis",
    "Creative writing",
    "Advice seeking",
    "Brainstorming",
    "Others",
]


class Classification(BaseModel):
    primary_tag: Tag
    other_tags: List[Tag]


def input_classification(inputs: str):
    return (
        f"""
# Instruction

Please label the task tags for the user query.

## User Query
```
{inputs}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
""",
        Classification,
    )


class QualityRating(BaseModel):
    explanation: str
    input_quality: Literal["very poor", "poor", "average", "good", "excellent"]


def input_quality_rating(inputs: str, answer: str):
    return (
        f"""
# Instruction

You need to rate the quality of the user query and answer based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is and answer unclear, vague, or incoherent. It lacks essential information and context. The query or answer includes non-Japanese nor English once or more.
- poor: The query and answer is somewhat unclear or lacks important details. It requires significant clarification. The query or answer includes non-Japanse nor English once or more.
- average: The query and answer is moderately clear and specific. It may require some additional information for a complete understanding. Also the query or answer is mostly in Japanese.
- good: The query and answer is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent. Its well formatted. Also the query or answer is mostly in Japanese.
- excellent: The query and answer is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response. Its pretty well formatted. Also the query or answer is mostly in Japanese.

Rate under poor when the query or answer includes non-Japanese nor English even one time.
Rate under poor if most of the query or answer is in English.

## User Query
```
{inputs}
```

## Answer
```
{answer}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
```
""",
        QualityRating,
    )


class PoClassify(BaseModel):
    explanation: str
    better: Literal["first", "second"]


def input_po_classification(
    inputs: str, answer1: str, answer2: str, rating1: str, rating2: str
):
    return (
        f"""# Instruction

You need to rate which output is better based on the inputs.
You have to provide reasons why either is better and either is bad.

## User Query
```
{inputs}
```

## First Answer
```
{answer1}
```

## First Answer Rating
```
{rating1}
```

## Second Answer
```
{answer2}
```

## Second Answer Rating
```
{rating2}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the answer.
Then, you need to output a rating which is better by filling in the placeholders in [...]:
```
{{
    "explanation": "[...]",
    "better": "[first/second]"
}}
```
""",
        PoClassify,
    )
