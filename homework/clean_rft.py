import json
from pathlib import Path

from data import is_answer_valid
from base_llm import BaseLLM


def clean_rft(input_path="../data/rft.json", output_path="../data/rft_clean.json"):
    parser = BaseLLM()

    with Path(input_path).open() as f:
        data = json.load(f)

    cleaned = []
    for row in data:
        if len(row) != 3:
            continue

        question, correct_answer, reasoning = row
        parsed = parser.parse_answer(reasoning)
        text = reasoning.strip()

        if parsed != parsed:
            continue
        if not is_answer_valid(parsed, correct_answer):
            continue
        if text.count("<answer>") != 1 or text.count("</answer>") != 1:
            continue
        if not text.endswith("</answer>"):
            continue
        if len(text) > 180:
            continue

        cleaned.append([question, correct_answer, text])

    with Path(output_path).open("w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"kept {len(cleaned)} / {len(data)}")


if __name__ == "__main__":
    clean_rft()
