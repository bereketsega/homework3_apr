from pathlib import Path
import torch

from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        from tqdm import tqdm

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size),
                desc=f"LLM Running on Micro Batches {micro_batch_size}",
            ):
                batch_result = self.batched_generate(
                    prompts[idx : idx + micro_batch_size],
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                )
                results.extend(batch_result)
            return results

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        nrs = 1 if num_return_sequences is None else num_return_sequences
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                num_return_sequences=nrs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_width = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_width:]
        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.tokenizer.padding_side = original_padding_side

        if num_return_sequences is None:
            return decoded

        return [decoded[i : i + nrs] for i in range(0, len(decoded), nrs)]


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    labels = [-100] * question_len + input_ids[question_len:]
    labels = labels[: len(input_ids)]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str, reasoning: str | None = None) -> dict[str, str]:
    if reasoning is not None:
        return {
            "question": prompt,
            "answer": reasoning.strip(),
        }

    rounded = round(float(answer), 3)
    return {
        "question": prompt,
        "answer": f"<answer>{rounded}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item) == 2:
            formated_data = self.format_fn(item[0], item[1])
        else:
            formated_data = self.format_fn(item[0], item[1], item[2])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = SFTLLM()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.config.use_cache = False
    llm.model.train()

    trainset = Dataset("train")
    tokenized_trainset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_trainset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTLLM()

    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
