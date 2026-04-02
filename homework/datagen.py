import json
from pathlib import Path

from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 6, temperature: float = 0.7):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset("train")
    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    rows = []
    pbar = tqdm(range(len(dataset)), desc="Generating RFT data")

    for i in pbar:
        question, correct_answer = dataset[i]
        prompt = model.format_prompt(question)

        candidates = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        chosen = None
        for candidate in candidates:
            parsed = model.parse_answer(candidate)
            cleaned = candidate.strip()
            if (
                parsed == parsed
                and is_answer_valid(parsed, correct_answer)
                and cleaned.count("<answer>") == 1
                and cleaned.count("</answer>") == 1
                and len(cleaned) < 200
            ):
                chosen = cleaned
                break

        if chosen is not None:
            rows.append([question, correct_answer, chosen])

        pbar.set_postfix(kept=len(rows))

        if (i + 1) % 10 == 0:
            with output_path.open("w") as f:
                json.dump(rows, f, indent=2)

    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"saved {len(rows)} samples to {output_path}", flush=True)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
