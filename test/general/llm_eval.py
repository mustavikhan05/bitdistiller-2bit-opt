import sys
from lm_eval import evaluator, tasks, utils
from utils_eval import LMEvalAdaptor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
import torch

sys.path.append("../")
from test_utils import pseudo_quantize_model_weight

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='Model to load; pass the location of the Hugging Face checkpoint.'
    )
    parser.add_argument('--eval_tasks', type=str, help='Evaluation tasks (e.g., hendrycksTest-*, arc_challenge, winogrande, hellaswag, piqa)')
    parser.add_argument('--test_set', action="store_true", help='Use the test set for evaluation')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--bits', type=int, default=2, help='Quantization bits')
    parser.add_argument('--group_size', type=int, default=128, help='Quantization group size')
    parser.add_argument('--quant_type', type=str, default="int", help='Quantization type')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples')
    args = parser.parse_args()
    print(args)

    if "hendrycksTest" not in args.eval_tasks:
        args.test_set = True

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # Use float16 for OPT models
        device_map='auto'
    )

    # Apply quantization if specified
    if args.quant_type is not None:
        q_config = {
            "zero_point": True,  # Default is True
            "q_group_size": args.group_size,  # Group quantization size
        }
        pseudo_quantize_model_weight(
            model, w_bit=args.bits, q_config=q_config, quant_type=args.quant_type
        )

    model.eval()

    # Load the tokenizer with the correct max_length and enable truncation
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        model_max_length=model.config.max_position_embeddings  # Set to match the model's capacity
    )

    # Set truncation parameters
    tokenizer.model_max_length = model.config.max_position_embeddings  # Ensure tokenizer knows the correct max length
    tokenizer.truncation_side = 'left'  # Truncate from the beginning if necessary

    # Print model and tokenizer max lengths for verification (optional)
    print(f"Model's maximum sequence length: {model.config.max_position_embeddings}")
    print(f"Tokenizer's maximum sequence length: {tokenizer.model_max_length}")

    task_names = utils.pattern_match(args.eval_tasks.split(","), tasks.ALL_TASKS)

    # Create an instance of the LMEvalAdaptor with the correct max_length
    lm_eval_model = LMEvalAdaptor(
        args.model,
        model,
        tokenizer,
        args.batch_size,
        max_length=model.config.max_position_embeddings
    )

    # Run evaluation without passing invalid arguments
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
        test_set=args.test_set
    )
    print(results)

    # Compute average accuracy
    acc_sum = 0
    count = 0
    if "hendrycksTest" in args.eval_tasks:
        for key in results['results']:
            if 'hendrycksTest' in key:
                acc_sum += results['results'][key]['acc']
                count += 1

        if count > 0:
            avg_acc = acc_sum / count
            mmlu_results = {'mmlu-acc': avg_acc}
            print(mmlu_results)
    else:
        for key in results['results']:
            acc_sum += results['results'][key]['acc']
            count += 1
        if count > 0:
            avg_acc = acc_sum / count
            print("QA Avg:", avg_acc)