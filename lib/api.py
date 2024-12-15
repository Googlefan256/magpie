from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from copy import deepcopy
from uuid import uuid4
from pydantic import BaseModel
from .inst import (
    PoClassify,
    QualityRating,
    Classification,
    DifficultyRating,
    input_classification,
    input_difficulty_rating,
    input_po_classification,
    input_quality_rating,
)

load_dotenv()
ai = OpenAI(base_url=os.environ["OPENAI_API_BASE"], api_key="hello")
tok: AutoTokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL"])
model = os.environ.get("MODEL_NAME") or "default"


def __magpie_schema(msg: str, js: BaseModel):
    c = (
        ai.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert AI."},
                {"role": "user", "content": msg},
            ],
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": js.model_json_schema()},
            },
            max_tokens=2048,
        )
        .choices[0]
        .message.content
    )
    return js.model_validate_json(c)


text = "cm20invba93"


def __magpie_generate(history: List, role: str):
    history = deepcopy(history)
    history.append({"role": role, "content": text})
    comp = tok.apply_chat_template(
        history, tokenize=False, add_special_tokens=True, add_generation_prompt=False
    ).split(text)[0]
    return (
        ai.completions.create(model=model, prompt=comp, max_tokens=2048).choices[0].text
    )


def magpie_conversation(system: str):
    i = str(uuid4())
    print(f"{i}: Started")
    try:
        history = [{"role": "system", "content": system}]
        user = __magpie_generate(history, "user")
        print(f"{i}: Generated user instruction")
        history = [{"role": "user", "content": user}]
        assistant = __magpie_generate(history, "assistant")
        print(f"{i} Generated answer")
        history.append({"role": "assistant", "content": assistant})
        rating: QualityRating = __magpie_schema(*input_quality_rating(user, assistant))
        if rating.input_quality in ["very poor", "poor"]:
            print(f"{i}: Quality {rating.input_quality} / {rating.explanation}")
            return None
        print(f"{i} Generated quality")
        classes: Classification = __magpie_schema(*input_classification(user))
        print(f"{i} Generated classify")
        difficulty: DifficultyRating = __magpie_schema(*input_difficulty_rating(user))
        print(f"{i}: Generated difficulty")
        return {
            "conversations": history,
            "quality": rating.input_quality,
            "rating_explanation": rating.explanation,
            "difficulty": difficulty.difficluty,
            "intent": difficulty.intent,
            "knowledge": difficulty.knowledge,
            "tag": classes.primary_tag,
            "tags": classes.other_tags,
            "system": system,
        }
    except Exception as e:
        print(f"{i}: {e}")
        return None


def magpie_preference(system: str):
    i = str(uuid4())
    print(f"{i}: Started")
    try:
        history = [{"role": "system", "content": system}]
        user = __magpie_generate(history, "user")
        print(f"{i}: Generated user instruction")
        history = [{"role": "user", "content": user}]
        assistant1 = __magpie_generate(history, "assistant")
        rating1: QualityRating = __magpie_schema(
            *input_quality_rating(user, assistant1)
        )
        assistant2 = __magpie_generate(history, "assistant")
        rating2: QualityRating = __magpie_schema(
            *input_quality_rating(user, assistant2)
        )
        if rating1.input_quality in [
            "very poor",
            "poor",
            "average",
        ] or rating2.input_quality in ["very poor", "poor", "average"]:
            print(f"{i}: Either bad quality")
            return None
        selection: PoClassify = __magpie_schema(
            *input_po_classification(
                user, assistant1, assistant2, rating1.explanation, rating2.explanation
            )
        )
        return {
            "was": selection.better,
            "explanation": selection.explanation,
            "selected_explanation": (
                rating1.explanation
                if selection.better == "first"
                else rating2.explanation
            ),
            "selected": [
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": (
                        assistant1 if selection.better == "first" else assistant2
                    ),
                },
            ],
            "rejected_explanation": (
                rating2.explanation
                if selection.better == "first"
                else rating1.explanation
            ),
            "rejected": [
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": (
                        assistant2 if selection.better == "first" else assistant1
                    ),
                },
            ],
            "prompt": user,
        }
    except Exception as e:
        print(f"{i}: {e}")
        return None
