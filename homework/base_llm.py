from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.eval()
        self.device = device

    def format_prompt(self, question: str) -> str:
        return question

    def parse_answer(self, answer: str) -> float:
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
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

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        pass

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        pass

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
                max_new_tokens=8,
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

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
