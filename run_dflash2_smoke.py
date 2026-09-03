import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash2 import dflash2_generate
from model import DFlash2DraftModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify lossless greedy DFlash2 generation against Qwen3."
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--draft-name-or-path",
        default="mgoin/Qwen3-4B-speculator.dflash2",
    )
    parser.add_argument(
        "--prompt",
        default="What is 17 multiplied by 24? Explain your reasoning.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--target-attn-implementation",
        default="sdpa",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DFlash2 smoke generation requires a CUDA GPU")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()
    draft = DFlash2DraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
    ).to(device)
    stop_token_ids = [tokenizer.eos_token_id]

    baseline_output_ids = target.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    speculative = dflash2_generate(
        model=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids,
    )

    if not torch.equal(
        baseline_output_ids.cpu(),
        speculative.output_ids,
    ):
        raise RuntimeError(
            "DFlash2 output differs from greedy Qwen3-4B output; "
            "lossless generation check failed"
        )

    generated_ids = speculative.output_ids[
        0,
        speculative.num_input_tokens :,
    ]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print()
    print("Lossless token match: PASS")
    print(f"Generated tokens: {speculative.num_output_tokens}")
    print(f"Decode rounds: {speculative.decode_rounds}")
    if speculative.acceptance_lengths:
        mean_acceptance = sum(speculative.acceptance_lengths) / len(
            speculative.acceptance_lengths
        )
        print(f"Mean accepted tokens per round: {mean_acceptance:.2f}")
    print(
        "Checkpoint caveat: experimental Speculators-trained DFlash2 "
        "checkpoint; not an official reproduction of Inco's unpublished "
        "training recipe."
    )


if __name__ == "__main__":
    main()
