import os
from typing import Dict, List, Union
import json
from utils.utils import (
    count_predictions,
    get_labels,
    get_task_labels,
    check_labels_have_punctuation,
)
import numpy as np
import argparse
from nmtscore import NMTScorer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)).item()


def inference_step(
    jsonl_path: str,
    output_path: str,
    model_name_or_path: str,
    source_lang: str,
    target_lang: str,
    normalize: bool = False,
    both_directions: bool = False,
    top_k: int = None,
    method: str = "score_direct",
    use_cpu: bool = False,
    agglutinative_language: bool = False,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    scorer = NMTScorer(model_name_or_path, device="cuda:0" if not use_cpu else None)

    print(f"Loading {jsonl_path}.\n")

    task_labels = get_task_labels(jsonl_path)
    labels_have_punctuation = check_labels_have_punctuation(jsonl_path)

    data_dictionary: Dict[str, Dict[str, Union[float, None]]] = {
        "#@!@#_model_name": model_name_or_path
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words_a: List[str] = []
    words_b: List[str] = []

    for line in lines:
        json_dict: Dict[str, Union[str, List[str]]] = json.loads(line)
        source_labels: str = json_dict["source_entity"].strip()
        predictions: List[str] = json_dict["preds"]
        target_sentence: str = json_dict["target_word"].strip()
        target_words: List[str] = target_sentence.split()

        # print(f"target_sentence: {target_sentence}")
        # print(f"Predictions: {predictions}")

        prediction_counter: Dict[str, Dict[str, int]] = count_predictions(
            predictions=predictions,
            target_words=target_words,
            top_k=top_k,
            task_labels=task_labels,
            labels_have_punctuation=labels_have_punctuation,
            agglutinative_language=agglutinative_language,
        )

        # print(f"prediction_counter: {prediction_counter}")

        source_labels, source_label_types = get_labels(
            source_labels,
            possible_labels=task_labels,
            labels_have_punctuation=labels_have_punctuation,
        )

        # words_a_temp = []
        # words_b_temp = []
        for text, label_class in zip(source_labels, source_label_types):
            text = " ".join(text.strip().split())
            if text not in data_dictionary:
                data_dictionary[text] = {}
            if label_class in prediction_counter:
                for word in prediction_counter[label_class].keys():
                    word = " ".join(word.strip().split())
                    data_dictionary[text][word] = None
                    words_a.append(text)
                    words_b.append(word)
                    # words_a_temp.append(text)
                    # words_b_temp.append(word)
        # print(f"words_a: {words_a_temp}")
        # print(f"words_b: {words_b_temp}")
        # print("\n\n")

    print(f"Building dictionary of {len(words_a)} pairs")
    if method == "score_direct":
        scores = scorer.score_direct(
            a=words_a,
            b=words_b,
            a_lang=source_lang,
            b_lang=target_lang,
            normalize=normalize,
            both_directions=both_directions,
        )
    elif method == "score_cross_likelihood":
        scores = scorer.score_cross_likelihood(
            words_a,
            words_b,
            a_lang=source_lang,
            b_lang=target_lang,
            tgt_lang=target_lang,
            normalize=normalize,
            both_directions=both_directions,
        )
    elif method == "score_pivot":
        scores = scorer.score_pivot(
            words_a,
            words_b,
            a_lang=source_lang,
            b_lang=target_lang,
            pivot_lang="fr" if "nllb" not in model_name_or_path else "fra_Latn",
            normalize=normalize,
            both_directions=both_directions,
        )
    else:
        raise ValueError(f"Method {method} not supported.")

    print(f"Update dictionary")
    for i, (word_a, word_b) in enumerate(zip(words_a, words_b)):
        data_dictionary[word_a][word_b] = scores[i]

    print(f"Saving results to {output_path}.\n")
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(data_dictionary, fp=f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the jsonl file containing the T5 outputs",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to store the computed scores",
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the scores",
    )

    parser.add_argument(
        "--both_directions",
        action="store_true",
        help="Compute scores for both directions",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top translation candidates to consider. None means all candidates",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="score_direct",
        help="Method to use for computing the scores",
        choices=["score_direct", "score_cross_likelihood", "score_pivot"],
    )

    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use CPU instead of GPU",
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

    inference_step(
        jsonl_path=args.jsonl_path,
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        normalize=args.normalize,
        both_directions=args.both_directions,
        top_k=args.top_k,
        method=args.method,
        use_cpu=args.use_cpu,
        agglutinative_language=args.agglutinative_language,
    )
