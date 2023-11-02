import os
from typing import List, Dict, Union
from tqdm import tqdm
import json
import argparse
from utils.baselines import evaluate_tsv

from utils.utils import (
    count_predictions,
    get_labels,
    subfinder,
    read_all_sentences_tsv,
    get_task_labels,
    check_labels_have_punctuation,
)
import torch
import string
from transformers import PreTrainedTokenizerBase, AutoTokenizer


def find_dictionary(
    dictionary: Dict[str, Dict[str, float]],
    source_txt: str,
    target_txt: str,
    tokenizer: PreTrainedTokenizerBase,
):
    original_source = source_txt
    original_target = target_txt
    try:  # Try to find the exact match
        s = dictionary[source_txt]
    except KeyError:
        try:  # Remove punctuation and try again
            source_txt = source_txt.translate(str.maketrans("", "", string.punctuation))
            s = dictionary[source_txt]
        except KeyError:
            try:  # Sometimes the tokenizer changes some characters, por example (1º->1o),
                # so we encode and decode the text to see if the tokenizer changed something
                tokens = tokenizer.encode(source_txt, add_special_tokens=False)
                source_txt = tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                s = dictionary[source_txt]
            except KeyError:
                raise KeyError(
                    f"Dictionary[{original_source}] not found! Something went wrong!. "
                    f"Clean string: {source_txt}.from"
                )
    try:
        t = s[target_txt]
    except KeyError:
        try:
            target_txt = target_txt.translate(str.maketrans("", "", string.punctuation))
            t = s[target_txt]
        except KeyError:
            try:
                tokens = tokenizer.encode(target_txt, add_special_tokens=False)
                target_txt = tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                t = s[target_txt]
            except KeyError:
                raise KeyError(
                    f"Dictionary[{source_txt}][{original_target}] not found! Something went wrong!. "
                    f"Clean string: {target_txt}"
                )

    return t


def get_score(
    source_txt: str,
    target_txt: str,
    dictionary: Dict[str, Dict[str, float]],
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    source_txt = source_txt.strip()
    target_txt = target_txt.strip()

    # Remove multiple spaces
    source_txt = " ".join(source_txt.split())
    target_txt = " ".join(target_txt.split())

    score = find_dictionary(
        dictionary=dictionary,
        source_txt=source_txt,
        target_txt=target_txt,
        tokenizer=tokenizer,
    )

    return score


def projection_step(
    jsonl_path: str,
    output_path: str,
    dictionary_path: str,
    top_k: int = None,
    gold_tsv: str = None,
    agglutinative_language: bool = False,
) -> Union[float, None]:
    print(f"==================== PROJECTION STEP ====================")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    task_labels = get_task_labels(jsonl_path)
    labels_have_punctuation = check_labels_have_punctuation(jsonl_path)

    with open(dictionary_path, "r", encoding="utf-8") as f:
        dictionary: Dict[str, Dict[str, float]] = json.load(f)

    if "#@!@#_model_name" in dictionary:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            dictionary["#@!@#_model_name"]
        )
    else:
        print(
            f"Dictionary does not contain model name. Using default tokenizer: facebook/m2m100_418M."
        )
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "facebook/m2m100_418M"
        )

    with open(jsonl_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    if gold_tsv is not None:
        sentences_words, _, _, _ = read_all_sentences_tsv(gold_tsv)
    else:
        sentences_words = None

    with open(output_path, "w", encoding="utf8") as output_file:
        for line_no, line in enumerate(
            tqdm(lines, desc="Projecting labels", ascii=True, leave=True)
        ):
            json_dict: Dict[str, Union[str, List[str]]] = json.loads(line.strip())
            source_labels: str = json_dict["source_entity"].strip()
            predictions: List[str] = json_dict["preds"]
            target_sentence: str = json_dict["target_word"].strip()
            target_words: List[str] = target_sentence.split()
            tags: List[str] = ["O"] * len(target_words)

            source_labels_words, source_label_types = get_labels(
                source_labels,
                possible_labels=task_labels,
                labels_have_punctuation=labels_have_punctuation,
            )

            prediction_counter: Dict[str, Dict[str, int]] = count_predictions(
                predictions=predictions,
                target_words=target_words,
                top_k=top_k,
                task_labels=source_label_types,
                labels_have_punctuation=labels_have_punctuation,
                agglutinative_language=agglutinative_language,
            )

            for source_txt, source_class in zip(
                source_labels_words, source_label_types
            ):
                try:
                    possible_targets = list(prediction_counter[source_class].keys())
                except KeyError:
                    continue

                # REMOVE OVERLAPPING
                for i in range(len(possible_targets) - 1, -1, -1):
                    matches = subfinder(
                        target_words,
                        possible_targets[i].split(),
                        tags,
                        agglutinative_language=agglutinative_language,
                    )
                    if len(matches) == 0:
                        del possible_targets[i]

                if len(possible_targets) == 0:
                    continue

                # Choose best candidate
                max_idx = torch.argmax(
                    torch.tensor(
                        [
                            get_score(
                                source_txt,
                                target_txt,
                                dictionary,
                                tokenizer,
                            )
                            for target_txt in possible_targets
                        ]
                    )
                )

                # Write tag

                matches = subfinder(
                    target_words,
                    possible_targets[max_idx].split(),
                    tags,
                    agglutinative_language=agglutinative_language,
                )

                idx = matches[0]
                tags[idx] = f"B-{source_class}"
                for x in range(
                    idx + 1,
                    min(idx + len(possible_targets[max_idx].split()), len(tags)),
                ):
                    tags[x] = f"I-{source_class}"

            for word, tag in zip(
                target_words if sentences_words is None else sentences_words[line_no],
                tags,
            ):
                print(f"{word} {tag}", file=output_file)
            print(file=output_file)

    if gold_tsv is not None:
        f1 = evaluate_tsv(
            original_dataset_path=gold_tsv,
            preds_path=output_path,
            output_dir=os.path.dirname(os.path.abspath(output_path)),
            output_name=os.path.splitext(os.path.basename(output_path))[0],
        )

        print(f"{os.path.splitext(os.path.basename(output_path))[0]} F1: {f1}")

        return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the jsonl file containing the T5 outputs",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output tsv file",
    )

    parser.add_argument(
        "--dictionary_path",
        type=str,
        required=True,
        help="Path to the dictionary file (json) from calculate_scores_nmts.py",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Use only the top k candidates",
    )

    parser.add_argument(
        "--gold_tsv",
        type=str,
        default=None,
        help="Path to the gold tsv file to evaluate the results. If not provided, no evaluation will be performed",
    )

    parser.add_argument(
        "--agglutinative_language",
        action="store_true",
        help="The target language is agglutinative.\n If you set the flag, "
        "Given the following sentence: Ummeli waseJamani kwikomiti yezilwanyana yeManyano yaseYurophu... \n"
        "And the following Location: ['Jamanai']. We will annotate waseJamani as Location.\n"
        "If you don't set the flag, we will force exact mathing and we will consider 'Jamanai' as an hallucination, "
        "because the exact word does not appear in the sentence, so we will discard it and we won't "
        "annotate anything.\n"
        "This flag is useful when projecting from a non-agglutinative language to an agglutinative language. "
        "For example, from English to Zulu or Xhosa. Please, before using this flag, check the annotations and "
        "guidelines of the dataset you are using, to ensure that this is the correct behaviour. You might want to "
        "perform a pre-processing tokenization step depending on the dataset and language you are using, for example, "
        "this flag won't work with Chinese or Japanese unless you perform a pre-processing tokenization step.",
    )

    args = parser.parse_args()

    projection_step(
        jsonl_path=args.jsonl_path,
        output_path=args.output_path,
        dictionary_path=args.dictionary_path,
        top_k=args.top_k,
        gold_tsv=args.gold_tsv,
        agglutinative_language=args.agglutinative_language,
    )
