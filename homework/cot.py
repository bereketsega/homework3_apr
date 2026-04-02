import torch

from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(checkpoint=checkpoint)

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You solve unit conversion problems. "
                    "Be concise and accurate. "
                    "Use one short calculation. "
                    "Every response must contain exactly one <answer>number</answer> block. "
                    "Put only the number inside the answer tags. "
                    "Do not include units inside the answer tags. "
                    "Do not skip the answer tags."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Convert 2 hours to minutes. "
                    "Write one short calculation and then the final answer."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "1 hour = 60 minutes. 2 x 60 = 120. "
                    "<answer>120</answer>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question} "
                    "Write one short calculation and then the final answer as "
                    "<answer>number</answer>."
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str) -> str:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
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
                max_new_tokens=64,
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


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})