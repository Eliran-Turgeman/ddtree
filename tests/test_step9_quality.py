from types import SimpleNamespace

import pytest
import torch

from evaluate_step62_quality import (
    benchmark_user_content,
    math_gold_answer,
    method_identity,
    validate_saved_prompt_hashes,
)
from summarize_evalplus_results import method_identity as evalplus_identity


@pytest.mark.parametrize(
    ("method_key", "expected"),
    [
        ("baseline", ("target-only", None)),
        ("dflash", ("dflash", None)),
        ("dflash2", ("dflash2", None)),
        ("ddtree_tb16", ("dflash-original-ddtree", 16)),
        (
            "dflash2_original_ddtree_tb32",
            ("dflash2-original-ddtree", 32),
        ),
        ("dflash2_unary_k16_tb64", ("unary", 64)),
        ("dflash2_pairwise_k16_tb7", ("pairwise", 7)),
    ],
)
def test_step9_quality_method_identity(method_key, expected) -> None:
    assert method_identity(method_key) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("baseline-sanitized_eval_results", ("target-only", None)),
        ("dflash-sanitized_eval_results", ("dflash", None)),
        ("dflash2-sanitized_eval_results", ("dflash2", None)),
        (
            "ddtree_tb16-sanitized_eval_results",
            ("dflash-original-ddtree", 16),
        ),
        (
            "dflash2_original_ddtree_tb32-sanitized_eval_results",
            ("dflash2-original-ddtree", 32),
        ),
        (
            "dflash2_unary_k16_tb64-sanitized_eval_results",
            ("unary", 64),
        ),
        (
            "dflash2_pairwise_k16_tb7-sanitized_eval_results",
            ("pairwise", 7),
        ),
    ],
)
def test_step9_evalplus_method_identity(stem, expected) -> None:
    assert evalplus_identity(stem) == expected


def test_gsm8k_gold_answer_extracts_final_numeric_answer() -> None:
    sample = {
        "answer": "Compute the quantity step by step.\n#### 1,234",
    }

    assert math_gold_answer("gsm8k", sample) == "1234"


def test_math500_gold_answer_is_already_canonical() -> None:
    assert math_gold_answer("math500", {"answer": r"\frac{1}{2}"}) == r"\frac{1}{2}"


def test_prompt_validation_falls_back_to_saved_input_tokens() -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return "|".join(message["content"] for message in messages)

        @staticmethod
        def encode(text, return_tensors):
            assert return_tensors == "pt"
            return torch.tensor([[ord(character) for character in text]])

    dataset = [{"question": "How many?", "answer": "#### 4"}]
    input_text = benchmark_user_content("gsm8k", dataset[0])
    input_ids = Tokenizer.encode(input_text, return_tensors="pt")
    result = SimpleNamespace(
        output_ids=torch.cat((input_ids, torch.tensor([[1, 2]])), dim=1),
        num_input_tokens=input_ids.shape[1],
    )
    run = {"responses": [{"baseline": result, "dflash": result}]}

    validated = validate_saved_prompt_hashes(run, "gsm8k", dataset, Tokenizer())

    assert validated == 2
