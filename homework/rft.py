from pathlib import Path
import json
import torch

from peft import PeftModel

from .base_llm import BaseLLM
from .sft import TokenizedDataset, format_example
from .data import Dataset, benchmark


class RFTLLM(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(checkpoint=checkpoint)

    def generate(self, prompt: str) -> str:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
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
                max_new_tokens=48,
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


class RFTDataset:
    def __init__(self, path: str):
        with Path(path).open() as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def load():
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = RFTLLM()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.config.use_cache = False
    llm.model.train()

    data_path = Path(__file__).parent.parent / "data" / "rft.json"
    trainset = RFTDataset(str(data_path))
    tokenized_trainset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=2e-4,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        save_strategy="no",
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


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = RFTLLM()

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
