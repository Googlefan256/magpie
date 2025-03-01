from openai import OpenAI
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("./output/sft-checkpoint/checkpoint-970")
model = AutoModelForCausalLM.from_pretrained(
    "./output/sft-checkpoint/checkpoint-970",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model = torch.compile(
    model, options={"triton.cudagraphs": True}, fullgraph=True, dynamic=True
)

ai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai")


def generate_response(q: str):
    inputs = tokenizer(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": q},
            ],
            add_generation_prompt=True,
            tokenize=False,
        ),
        return_tensors="pt",
    ).to(device)
    generated = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=2000,
            num_beams=1,
            do_sample=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|end_of_turn|>"),
        ),
    )
    return tokenizer.decode(generated.squeeze(0)[inputs.input_ids.size(-1) : -1])


def generate_score(p: str):
    try:
        res = ai.chat.completions.create(
            model="gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": p}],
            extra_body={},
        )
        return float(res.choices[0].message.content.strip())
    except Exception as e:
        time.sleep(2)
        print(f"Error: {e}")
        return generate_score(p)


def eval_one(q: str, a: str, aspect: str):
    pred = generate_response(q)
    res = generate_score(
        f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{q}

# 正解例
{a}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{aspect}

# 回答
{pred}"""
    )
    return res, pred


if __name__ == "__main__":
    ds = load_dataset("elyza/ELYZA-tasks-100", split="test")
    score = 0.0
    res = []
    pbar = tqdm(ds)
    for i, entry in enumerate(pbar):
        q = entry["input"]
        a = entry["output"]
        aspect = entry["eval_aspect"]
        s, pred = eval_one(
            q,
            a,
            aspect,
        )
        score += s
        pbar.set_description(f"Avg. {score / (i + 1)}")
        print(f"Score: {s}\nQuestion: {q}\nAnswer: {a}\nPred: {pred}")
        res.append(
            {"score": int(s), "question": q, "pure_answer": a, "pred_answer": pred}
        )
    print(f"Score: {score / len(ds)}")
