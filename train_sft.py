from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

proc = os.cpu_count() * 3 // 4
ds = load_dataset("usable-japanese-llm/magpie-sft-jp", split="train")

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="sbintuitions/sarashina2.1-1b",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
)
tokenizer.add_tokens(
    [
        "<|start_of_turn|>",
        "<|end_of_turn|>",
        "<|user|>",
        "<|assistant|>",
    ]
)
model.resize_token_embeddings(len(tokenizer))
tokenizer.chat_template = open("./template.jinja").read()


def formatting(batch):
    return {
        "text": tokenizer.apply_chat_template(
            [
                conv
                for i, conv in enumerate(batch["conversations"])
                if batch["quality"][i] in ["good", "excellent"]
            ],
            tokenize=False,
            add_special_tokens=True,
            add_generation_prompt=False,
        )
    }


ds = ds.map(formatting, batched=True, num_proc=proc, remove_columns=ds.column_names)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="./output/sft-checkpoint",
        optim="adamw_8bit",
        learning_rate=2e-5,
        lr_scheduler_type="constant_with_warmup",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=2,
    ),
)
trainer.train()
trainer.save_pretrained("./output/sft-final")
