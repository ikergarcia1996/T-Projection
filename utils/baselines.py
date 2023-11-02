from typing import List, Dict, Union
import json
from tqdm.auto import tqdm
from utils.utils import (
    count_predictions,
    subfinder,
    get_labels,
    get_task_labels,
    check_labels_have_punctuation,
)
import os
import argparse
from utils.tag_encoding import rewrite_labels
from utils.utils import read_all_sentences_tsv
from seqeval.metrics import f1_score, classification_report


def evaluate_tsv(
    original_dataset_path: str,
    preds_path: str,
    output_dir: str,
    output_name: str,
):
    _, gold_labels, _, _ = read_all_sentences_tsv(original_dataset_path)
    _, predicted_labels, _, _ = read_all_sentences_tsv(preds_path)

    gold_labels = [rewrite_labels(tags, encoding="iob2") for tags in gold_labels]
    predicted_labels = [
        rewrite_labels(tags, encoding="iob2") for tags in predicted_labels
    ]

    with open(
        os.path.join(output_dir, output_name + ".scores"), "w", encoding="utf8"
    ) as output_file:
        try:
            cr = classification_report(
                y_true=gold_labels, y_pred=predicted_labels, digits=4, zero_division="1"
            )
        except ValueError as e:
            cr = str(e)
        print(
            cr,
            file=output_file,
        )
        try:
            micro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating micro f1: {e}")
            micro_f1 = 0

        try:
            macro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating macro f1: {e}")
            macro_f1 = 0
        print(f"Micro F1: {micro_f1}", file=output_file)
        print(f"Macro F1: {macro_f1}", file=output_file)

    return micro_f1


def write_tags(
    counter: Dict[
        str,
        Dict[str, int],
    ],
    target_words: List[str],
    tags: List[str],
) -> List[str]:
    for source_class, target_dict in counter.items():
        for target_txt, _ in target_dict.items():
            matches = subfinder(target_words, target_txt.split(), tags)
            try:
                idx = matches[0]
                tags[idx] = "B-" + source_class
                for i in range(idx + 1, idx + len(target_txt.split())):
                    tags[i] = "I-" + source_class
            except IndexError:
                pass
    return tags


def evaluate_most_probable(
    jsonl_path,
    output_path: str,
    gold_tsv: str = None,
) -> Union[float, None]:
    # We only use the most probable model prediction
    with open(jsonl_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    task_labels = get_task_labels(jsonl_path)
    labels_have_punctuation = check_labels_have_punctuation(jsonl_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as output_file:
        for line in tqdm(lines, desc="Projecting labels", ascii=True, leave=True):
            json_dict: Dict[str, Union[str, List[str]]] = json.loads(line.strip())
            predictions: List[str] = json_dict["preds"]
            target_sentence: str = json_dict["target_word"].strip()
            target_words: List[str] = target_sentence.split()
            tags: List[str] = ["O"] * len(target_words)
            source_entity: str = json_dict["source_entity"].strip()
            source_entity_txt, source_entity_class = get_labels(
                source_entity,
                possible_labels=task_labels,
                labels_have_punctuation=labels_have_punctuation,
            )

            prediction_counter: Dict[str, Dict[str, int]] = count_predictions(
                predictions=predictions,
                target_words=target_words,
                top_k=1,
                task_labels=source_entity_class,  # We only consider classes that are in the source sentence
                labels_have_punctuation=labels_have_punctuation,
            )

            tags = write_tags(
                counter=prediction_counter, target_words=target_words, tags=tags
            )

            for word, tag in zip(target_words, tags):
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


def evaluate_best_prediction(
    jsonl_path,
    output_path: str,
    gold_tsv: str = None,
    top_k: int = None,
) -> Union[float, None]:
    # We test the F1 score of each prediction, we choose the prediction with the highest f1 score
    # IMPORTANT: This function uses the test data to select the best candidate, It is intended as an UPPERBOUND.
    #            DO NOT USE THIS FUNCTION TO REPORT RESULTS
    with open(jsonl_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    task_labels = get_task_labels(jsonl_path)
    labels_have_punctuation = check_labels_have_punctuation(jsonl_path)

    with open(output_path, "w", encoding="utf8") as output_file:
        for line in tqdm(lines, desc="Projecting labels", ascii=True, leave=True):
            json_dict: Dict[str, Union[str, List[str]]] = json.loads(line.strip())
            predictions: List[str] = json_dict["preds"]
            target_sentence: str = json_dict["target_word"].strip()
            target_entity: str = json_dict["target_entity"].strip()

            target_entity_txt, target_entity_class = get_labels(
                target_entity,
                possible_labels=task_labels,
                labels_have_punctuation=labels_have_punctuation,
            )

            target_words: List[str] = target_sentence.split()

            tags: List[str] = ["O"] * len(target_words)

            prediction_counter: Dict[str, Dict[str, int]] = count_predictions(
                predictions=predictions,
                target_words=target_words,
                top_k=top_k,
                task_labels=target_entity_class,  # We only consider classes that are in the source sentence
                labels_have_punctuation=labels_have_punctuation,
            )

            for label_words, label_class in zip(target_entity_txt, target_entity_class):
                if label_class in prediction_counter:
                    if label_words in prediction_counter[label_class]:
                        matches = subfinder(target_words, label_words.split(), tags)
                        try:
                            idx = matches[0]
                            tags[idx] = "B-" + label_class
                            for i in range(idx + 1, idx + len(label_words.split())):
                                tags[i] = "I-" + label_class
                        except IndexError:
                            print(
                                f"WARNING: Gold label {label_words} not found in {target_sentence}\n"
                                f"json_dict: {json_dict}\n"
                            )
                    # else:
                    #    print(
                    #        f"WARNING: Gold label {label_words} not found in {target_sentence}\n"
                    #        f"prediction_counter: {prediction_counter}\n"
                    #        f"json_dict: {json_dict}\n"
                    #    )

            for word, tag in zip(target_words, tags):
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
        "--method",
        type=str,
        default="most_probable",
        choices=["most_probable", "best_prediction"],
        help="The method to use to project the labels",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the jsonl file containing the predictions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--gold_tsv",
        type=str,
        required=False,
        default=None,
        help="Path to the gold tsv file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=None,
        help="Number of predictions to consider",
    )

    args = parser.parse_args()

    if args.method == "most_probable":
        evaluate_most_probable(
            jsonl_path=args.jsonl_path,
            output_path=args.output_path,
            gold_tsv=args.gold_tsv,
        )
    elif args.method == "best_prediction":
        evaluate_best_prediction(
            jsonl_path=args.jsonl_path,
            output_path=args.output_path,
            gold_tsv=args.gold_tsv,
            top_k=args.top_k,
        )

    else:
        raise ValueError(f"Unknown method {args.method}")
