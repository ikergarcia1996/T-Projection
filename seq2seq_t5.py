import argparse
import math
import os
from utils.dataset import get_dataloader_t5
from utils.baselines import evaluate_most_probable
import json

from tqdm.auto import tqdm

import torch
from accelerate import Accelerator


from fairseq.optim.adafactor import Adafactor
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    PreTrainedTokenizer,
)

from utils.utils import get_tags


try:
    import wandb

    wandb.require("service")
except ImportError:
    wandb = None


def gen_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--train_tsv",
        type=str,
        default=None,
        help="A tsv file in conll format containing the sl training data.",
    )

    parser.add_argument(
        "--dev_tsv",
        type=str,
        default=None,
        help="A tsv file in conll format containing the dev data.",
    )

    parser.add_argument(
        "--test_source_tsv",
        nargs="+",
        type=str,
        default=None,
        help="A tsv file in conll format containing the source language test data.",
    )
    parser.add_argument(
        "--test_target_tsv",
        nargs="+",
        type=str,
        default=None,
        help="A tsv file in conll format containing the target language test data.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=15,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=15,
        help="Number of sequences to return. This argument will be "
        "passed to ``model.generate``, which is used during ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the inference dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate every epoch or every eval_every_epochs",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local trank for distributed training",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="T5-Sequence-Labeling",
        help="The project name to use for wandb.",
    )

    args = parser.parse_args()

    # Sanity checks

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_tsv is not None and args.dev_tsv is None:
        raise ValueError("You must specify a dev set if you specify a train set.")

    if args.test_source_tsv is not None:
        assert len(args.test_source_tsv) == len(args.test_target_tsv), (
            f"The number of test files should be the same for source, target "
            f"len test_source_tsv: {len(args.test_source_tsv)}. "
            f"len test_target_tsv: {len(args.test_target_tsv)}. "
        )

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.train_tsv is not None:
        # deepspeed_plugin = DeepSpeedPlugin(
        #    zero_stage=2,
        #    gradient_accumulation_steps=args.gradient_accumulation_steps,
        #    offload_optimizer_device="cpu",
        # )
        # accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
        accelerator = Accelerator()
    else:
        accelerator = Accelerator()

    # Load the model
    print(f"Loading config from {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, legacy=True
    )

    print(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    if "flan" in args.model_name_or_path:
        print(
            f"FlanT5 model detected, adding labels to tokenizer. "
            f"FlanT5 tokenizer doesn't play nice with the html-style tags, so "
            f"to prevent tokenization errors we add the labels to the tokenizer."
        )
        if args.train_tsv is not None:
            start_tags, end_tags = get_tags(args.train_tsv)
        else:
            start_tags, end_tags = [], []
            for test_test in args.test_source_tsv:
                s, t = get_tags(test_test)
                start_tags.extend(s)
                end_tags.extend(t)

        print(f"Len tokenizer before: {len(tokenizer)}")
        tokenizer.add_tokens(start_tags)
        tokenizer.add_tokens(end_tags)
        print(f"Len tokenizer after: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if args.train_tsv is not None:
        print(f"Loading training")
        train_dataloader = get_dataloader_t5(
            source_tsv_path=args.train_tsv,
            target_tsv_path=args.train_tsv,
            tokenizer=tokenizer,
            max_source_len=args.max_source_length,
            max_target_len=args.max_source_length,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )

        print(f"Loading validation data")
        val_dataloader = get_dataloader_t5(
            source_tsv_path=args.dev_tsv,
            target_tsv_path=args.dev_tsv,
            tokenizer=tokenizer,
            max_source_len=args.max_source_length,
            max_target_len=args.max_source_length,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
        )

    else:
        train_dataloader = None
        val_dataloader = None

    if train_dataloader is not None:
        wandb.init(
            project=args.project_name,
            name=f"{os.path.basename(args.output_dir)}",
            config={
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_source_length,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "output_dir": args.output_dir,
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_return_sequences,
                "local_rank": args.local_rank,
            },
        )

        wandb.config.model_name_or_path = args.model_name_or_path
        wandb.config.per_device_train_batch_size = args.per_device_train_batch_size
        wandb.config.gradient_accumulation_steps = args.gradient_accumulation_steps
        wandb.config.learning_rate = args.learning_rate
        wandb.config.weight_decay = args.weight_decay
        wandb.config.lr_scheduler_type = args.lr_scheduler_type
        wandb.config.num_warmup_steps = args.num_warmup_steps
        wandb.config.seed = args.seed
        wandb.config.eval_every = args.eval_every
        wandb.config.Mixed_precision = accelerator.mixed_precision
        wandb.config.Num_GPUs = accelerator.num_processes
        wandb.config.num_train_epochs = args.num_train_epochs

        print(f"Preparing for training...")

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader)
            / args.gradient_accumulation_steps
            / accelerator.num_processes
        )

        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        wandb.config.total_batch_size = total_batch_size
        wandb.config.max_train_steps = max_train_steps

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        """
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-04
        )
        """
        optimizer = Adafactor(
            params=optimizer_grouped_parameters,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=args.learning_rate,
            clip_threshold=1.0,
            # weight_decay=args.weight_decay,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        completed_steps = 0

        best_epoch: int = -1
        best_epoch_metric: float = -1
        validation_dir: str = os.path.join(args.output_dir, "val_logs")
        os.makedirs(validation_dir, exist_ok=True)

        running_loss = 0
        num_batches = 0

        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": 1,
            "num_return_sequences": 1,
        }

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataloader.dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total batch size = {total_batch_size}")
        print(f"  Total optimization steps = {max_train_steps}")
        print(f"  Learning rate = {args.learning_rate}")
        print(f"  Weight decay = {args.weight_decay}")
        print(f"  Scheduler = {args.lr_scheduler_type}")
        print(f"  Model = {args.model_name_or_path}")
        print(f"  Mixed Precision = {accelerator.mixed_precision}")
        print(f"  Local Rank = {args.local_rank}")
        print(f"  Accelerator State :\n{accelerator.state}\n")
        print()

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ascii=True,
            desc="Training",
        )

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, (
                model_inputs,
                task_name,
                idx,
                source_words,
                source_entities,
                source_labels,
                target_words,
                target_entities,
                target_labels,
            ) in enumerate(train_dataloader):
                outputs = model(**model_inputs)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                running_loss += loss.item()
                num_batches += 1

                wandb.log(
                    {
                        "Train/Loss": loss.item(),
                        "Train/epoch": epoch,
                        "Train/batch": num_batches,
                    }
                )

                wandb.log(
                    {
                        "Train/Running Loss": loss.item() / num_batches,
                        "Train/epoch": epoch,
                        "Train/batch": num_batches,
                    }
                )

                wandb.log(
                    {
                        "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                        "Train/epoch": epoch,
                        "Train/batch": num_batches,
                    }
                )

                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

            if (((epoch + 1) % args.eval_every) == 0) or (
                epoch + 1 == args.num_train_epochs
            ):
                if accelerator.is_local_main_process:
                    output_file = open(
                        os.path.join(validation_dir, f"dev_{epoch}.jsonl"),
                        "w",
                        encoding="utf8",
                    )

                    samples_seen: int = 0

                else:
                    output_file = None
                    samples_seen: int = 0

                print(f"***** Evaluating at epoch {epoch + 1} *****")
                print(f"  Num examples = {len(val_dataloader.dataset)}")
                print()

                model.eval()

                eval_progress_bar = tqdm(
                    range(len(val_dataloader)),
                    disable=not accelerator.is_local_main_process,
                    ascii=True,
                    desc="Validation",
                )

                for step, (
                    model_inputs,
                    task_name,
                    idx,
                    source_words,
                    source_entities,
                    source_labels,
                    target_words,
                    target_entities,
                    target_labels,
                ) in enumerate(val_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            model_inputs["input_ids"],
                            attention_mask=model_inputs["attention_mask"],
                            **gen_kwargs,
                        )
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        generated_tokens = (
                            accelerator.gather(generated_tokens).cpu().numpy()
                        )

                        source_words = accelerator.pad_across_processes(
                            source_words,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        source_words = accelerator.gather(source_words).cpu().numpy()

                        source_entities = accelerator.pad_across_processes(
                            source_entities,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        source_entities = (
                            accelerator.gather(source_entities).cpu().numpy()
                        )

                        source_labels = accelerator.pad_across_processes(
                            source_labels,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        source_labels = accelerator.gather(source_labels).cpu().numpy()

                        target_words = accelerator.pad_across_processes(
                            target_words,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        target_words = accelerator.gather(target_words).cpu().numpy()

                        target_entities = accelerator.pad_across_processes(
                            target_entities,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        target_entities = (
                            accelerator.gather(target_entities).cpu().numpy()
                        )

                        target_labels = accelerator.pad_across_processes(
                            target_labels,
                            dim=1,
                            pad_index=tokenizer.pad_token_id,
                        )
                        target_labels = accelerator.gather(target_labels).cpu().numpy()

                        task_name = accelerator.gather(task_name).cpu().numpy()

                        idx = accelerator.gather(idx)

                        task_name = tokenizer.batch_decode(
                            task_name,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        decoded_preds = tokenizer.batch_decode(
                            generated_tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        source_words = tokenizer.batch_decode(
                            source_words,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        source_entities = tokenizer.batch_decode(
                            source_entities,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        source_labels = tokenizer.batch_decode(
                            source_labels,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        target_words = tokenizer.batch_decode(
                            target_words,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        target_entities = tokenizer.batch_decode(
                            target_entities,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        target_labels = tokenizer.batch_decode(
                            target_labels,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        if accelerator.is_local_main_process:
                            if accelerator.num_processes > 1:
                                if step == len(val_dataloader) - 1:
                                    decoded_preds = decoded_preds[
                                        : (len(val_dataloader.dataset) - samples_seen)
                                        * args.num_return_sequences
                                    ]

                                    source_words = source_words[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    source_entities = source_entities[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    source_labels = source_labels[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    target_words = target_words[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    target_entities = target_entities[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    target_labels = target_labels[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    task_name = task_name[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]
                                    idx = idx[
                                        : len(val_dataloader.dataset) - samples_seen
                                    ]

                                else:
                                    samples_seen += len(source_words)

                            for (
                                i,
                                task,
                                preds,
                                source_word,
                                source_entity,
                                source_label,
                                target_word,
                                target_entity,
                                target_label,
                            ) in zip(
                                idx,
                                task_name,
                                gen_batch(decoded_preds, n=1),
                                source_words,
                                source_entities,
                                source_labels,
                                target_words,
                                target_entities,
                                target_labels,
                            ):
                                dictionary = {
                                    "id": int(i.item()),
                                    "task": task,
                                    "source_word": source_word,
                                    "source_entity": source_entity,
                                    "source_label": source_label,
                                    "target_word": target_word,
                                    "target_entity": target_entity,
                                    "target_label": target_label,
                                    "preds": preds,
                                }
                                # print(dictionary)
                                print(
                                    json.dumps(dictionary, ensure_ascii=False),
                                    file=output_file,
                                )

                    eval_progress_bar.update(1)

                eval_progress_bar.close()
                accelerator.wait_for_everyone()

                if accelerator.is_local_main_process:
                    output_file.close()

                    f1 = evaluate_most_probable(
                        jsonl_path=os.path.join(validation_dir, f"dev_{epoch}.jsonl"),
                        output_path=os.path.join(validation_dir, f"dev_{epoch}.tsv"),
                        gold_tsv=args.dev_tsv,
                    )

                    wandb.log(
                        {
                            f"VAL/F1": f1,
                            "epoch": epoch,
                        }
                    )

                    if (
                        (f1 > best_epoch_metric)
                        or (best_epoch_metric < 0)
                        or (math.isnan(best_epoch_metric))
                    ):
                        best_epoch_metric = f1
                        best_epoch = epoch
                        print(f"NEW BEST MODEL :) epoch {best_epoch}. F1: {f1}")

                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, save_function=accelerator.save
                        )
                        tokenizer.save_pretrained(args.output_dir)
                    else:
                        print(
                            f"Epoch {epoch} F1 {f1} worse than {best_epoch_metric}. :("
                        )

        progress_bar.close()

    if args.test_source_tsv is not None and args.test_target_tsv is not None:
        print(f"========= TESTING =========")
        if train_dataloader is not None:
            print(f"Loading best model from {args.output_dir}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.output_dir,
                config=config,
            )

        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
        }

        for source_test_file, target_test_file in zip(
            args.test_source_tsv,
            args.test_target_tsv,
        ):
            print(f"Running test for files: {source_test_file}  {target_test_file}")

            test_dataloader = get_dataloader_t5(
                source_tsv_path=source_test_file,
                target_tsv_path=target_test_file,
                tokenizer=tokenizer,
                max_source_len=args.max_source_length,
                max_target_len=args.max_source_length,
                batch_size=args.per_device_test_batch_size,
                shuffle=False,
                inference=True,
            )

            model, test_dataloader = accelerator.prepare(model, test_dataloader)

            test_name = os.path.basename(os.path.splitext(target_test_file)[0])

            if accelerator.is_local_main_process:
                output_file = open(
                    os.path.join(args.output_dir, f"{test_name}.jsonl"),
                    "w",
                    encoding="utf8",
                )

                samples_seen: int = 0

            else:
                output_file = None
                samples_seen: int = 0

            print(f"***** Inference *****")
            print(f"  Num examples = {len(test_dataloader.dataset)}")
            print(f"  Gen kwargs = {json.dumps(gen_kwargs, indent=2)}")
            print()

            model.eval()

            test_progress_bar = tqdm(
                range(len(test_dataloader)),
                disable=not accelerator.is_local_main_process,
                ascii=True,
                desc="Inference",
            )
            first_n = 0  # Disable debug

            for step, (
                model_inputs,
                task_name,
                idx,
                source_words,
                source_entities,
                source_labels,
                target_words,
                target_entities,
                target_labels,
            ) in enumerate(test_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                        **gen_kwargs,
                    )
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )

                    source_words = accelerator.pad_across_processes(
                        source_words,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    source_words = accelerator.gather(source_words).cpu().numpy()

                    source_entities = accelerator.pad_across_processes(
                        source_entities,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    source_entities = accelerator.gather(source_entities).cpu().numpy()

                    source_labels = accelerator.pad_across_processes(
                        source_labels,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    source_labels = accelerator.gather(source_labels).cpu().numpy()

                    target_words = accelerator.pad_across_processes(
                        target_words,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    target_words = accelerator.gather(target_words).cpu().numpy()

                    target_entities = accelerator.pad_across_processes(
                        target_entities,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    target_entities = accelerator.gather(target_entities).cpu().numpy()

                    target_labels = accelerator.pad_across_processes(
                        target_labels,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    target_labels = accelerator.gather(target_labels).cpu().numpy()

                    task_name = accelerator.gather(task_name).cpu().numpy()

                    idx = accelerator.gather(idx)

                    task_name = tokenizer.batch_decode(
                        task_name,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    source_words = tokenizer.batch_decode(
                        source_words,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    source_entities = tokenizer.batch_decode(
                        source_entities,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    source_labels = tokenizer.batch_decode(
                        source_labels,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    target_words = tokenizer.batch_decode(
                        target_words,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    target_entities = tokenizer.batch_decode(
                        target_entities,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    target_labels = tokenizer.batch_decode(
                        target_labels,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    if accelerator.is_local_main_process:
                        if first_n > 0:
                            print("***** DEBUG *****")
                            print(f"Model inputs: {model_inputs}")
                            print(f"Source words: {source_words}")
                            print(f"Source entities: {source_entities}")
                            print()
                            print(f"Target words: {target_words}")
                            print(f"Target entities: {target_entities}")
                            print()
                            print(f"Generated tokens: {generated_tokens}")
                            print(f"Decoded preds: {decoded_preds}")
                            print()
                            first_n -= 1

                        if accelerator.num_processes > 1:
                            if step == len(test_dataloader) - 1:
                                decoded_preds = decoded_preds[
                                    : (len(test_dataloader.dataset) - samples_seen)
                                    * args.num_return_sequences
                                ]

                                source_words = source_words[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                source_entities = source_entities[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                source_labels = source_labels[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                target_words = target_words[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                target_entities = target_entities[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                target_labels = target_labels[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                task_name = task_name[
                                    : len(test_dataloader.dataset) - samples_seen
                                ]
                                idx = idx[: len(test_dataloader.dataset) - samples_seen]

                            else:
                                samples_seen += len(source_words)

                        for (
                            i,
                            task,
                            preds,
                            source_word,
                            source_entity,
                            source_label,
                            target_word,
                            target_entity,
                            target_label,
                        ) in zip(
                            idx,
                            task_name,
                            gen_batch(decoded_preds, n=args.num_return_sequences),
                            source_words,
                            source_entities,
                            source_labels,
                            target_words,
                            target_entities,
                            target_labels,
                        ):
                            dictionary = {
                                "id": int(i.item()),
                                "task": task,
                                "source_word": source_word,
                                "source_entity": source_entity,
                                "source_label": source_label,
                                "target_word": target_word,
                                "target_entity": target_entity,
                                "target_label": target_label,
                                "preds": preds,
                            }
                            # print(dictionary)
                            print(
                                json.dumps(dictionary, ensure_ascii=False),
                                file=output_file,
                            )

                            if step % 100 == 0:
                                # flush the file every 100 steps, so we can see the progress
                                output_file.flush()

                test_progress_bar.update(1)

            test_progress_bar.close()
            accelerator.wait_for_everyone()

            if accelerator.is_local_main_process:
                output_file.close()


if __name__ == "__main__":
    main()
