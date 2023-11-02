from typing import List, Dict, Optional, TextIO
from utils.label_names import name2label
import string
import sys
from utils.label_names import label2name
from utils.tag_encoding import rewrite_labels
import re
import json


def get_labels_deprecated(sentence: str) -> (List[str], List[str]):
    """
    Input
    <Person>Obama</Person><Location>USA</Location>
    Output
    ['Obama', 'USA']
    ['PER', 'LOC']
    """
    labels: List[str] = []
    label_types: List[str] = []
    current_label: List[str] = []
    current_type: List[str] = []
    sentence: str = sentence.strip()
    inside_tag: bool = False
    i: int = 0

    while i < len(sentence):
        if sentence[i] == "<" and i + 1 < len(sentence) and sentence[i + 1] == "/":
            if len(current_label) > 0 and len(current_type) > 0:
                # Remove spaces in the label class
                pred_label = "".join("".join(current_type).split())
                # Remove multiple spaces in label
                txt = " ".join("".join(current_label).split())
                # Remove punctuation
                txt = txt.translate(str.maketrans("", "", string.punctuation))
                labels.append(txt)
                label_types.append(name2label(pred_label))
            current_label = []
            current_type = []
            inside_tag = False

            while i < len(sentence) and sentence[i] != ">":
                i += 1
        elif sentence[i] == "<":
            inside_tag = True
        elif sentence[i] == ">":
            inside_tag = False
        else:
            if inside_tag:
                current_type.append(sentence[i])
            else:
                current_label.append(sentence[i])

        i += 1

    return labels, label_types


def split_sentence(
    tag_regex,
    sentence: str,
    recursion_limit: int = 256,
) -> List[str]:
    sentence = sentence.strip().split()

    if recursion_limit == 0:
        return sentence

    new_sentence: List[str] = []

    for word in sentence:
        search_result = tag_regex.search(word)
        if search_result:
            span = search_result.span()

            l = word[: span[0]].strip()
            r = word[span[1] :].strip()
            t = word[span[0] : span[1]].strip()
            if l:
                new_sentence.extend(split_sentence(tag_regex, l, recursion_limit - 1))
            new_sentence.append(t)
            if r:
                new_sentence.extend(split_sentence(tag_regex, r, recursion_limit - 1))

        else:
            new_sentence.append(word)

    return new_sentence


def get_label_type(label: str) -> (str, bool):
    label = label.strip()
    is_start = not label.startswith("</")
    if is_start:
        label_type = name2label(label[1:-1])
    else:
        label_type = name2label(label[2:-1])

    return label_type, is_start


def get_labels(
    prediction: str, possible_labels: List[str], labels_have_punctuation: bool
) -> (List[str], List[str]):
    """
    Input
    <Person>Obama</Person> went to <Location>New York</Location> .
    Output
    ["Obama","New York"]
    """
    inside_tag: bool = False

    tag_regex = re.compile(
        f"</?({'|'.join([label2name(p) for p in possible_labels])})>"
    )
    # print(f"Possible labels: {possible_labels}")
    predicted_words = split_sentence(tag_regex, prediction)
    # print(f"Predicted words: {predicted_words}")

    first = True
    entities = []
    entities_types = []
    current_entity = []
    current_entity_type: str = ""

    for word in predicted_words:
        result = tag_regex.match(word)
        if result:
            label_type, is_start = get_label_type(word)
            if is_start:
                inside_tag = True
                current_entity_type = label_type
                first = True
            else:
                if (
                    inside_tag
                    and current_entity
                    and len(current_entity_type) > 0
                    and current_entity_type == label_type
                ):
                    # Remove spaces in the label class
                    pred_label = "".join("".join(current_entity_type).split())
                    # Remove multiple spaces in label
                    txt = " ".join(" ".join(current_entity).split())
                    # Remove punctuation
                    if not labels_have_punctuation:
                        txt = txt.translate(str.maketrans("", "", string.punctuation))
                    entities.append(txt)
                    entities_types.append(name2label(pred_label))

                inside_tag = False
                current_entity_type = ""
        else:
            if inside_tag:
                if first:
                    current_entity = [word]
                    first = False
                else:
                    current_entity.append(word)

    return entities, entities_types


def find_sublist(
    mylist: List[str],
    pattern: List[str],
    tags: Optional[List[str]] = None,
    agglutinative_language: bool = False,
) -> List[int]:
    matches: List[int] = []

    if len(pattern) == 0:
        return matches

    if not agglutinative_language:
        for i in range(min(len(mylist), sys.maxsize if tags is None else len(tags))):
            if (
                (
                    mylist[i] == pattern[0] and (tags is None or tags[i] == "O")
                )  # First word matches
                and mylist[i : i + len(pattern)] == pattern  # Rest of the words match
                and (
                    tags is None
                    or all(  # All tags are "O"
                        [
                            tags[j] == "O"
                            for j in range(i, min(i + len(pattern), len(tags)))
                        ]
                    )
                )
            ):
                matches.append(i)

    else:
        """
        Given the following sentence:
        Ummeli waseJamani kwikomiti yezilwanyana yeManyano yaseYurophu ...
        And the following pattern:
        ["Jamanai"]
        The function will return:
        [1]

        We accept a match if each word in the pattern is a substring of the corresponding word in the sentence.
        """
        for i in range(min(len(mylist), sys.maxsize if tags is None else len(tags))):
            if (
                (
                    pattern[0] in mylist[i] and (tags is None or tags[i] == "O")
                )  # First is the same
                and len(mylist[i : i + len(pattern)])
                == len(pattern)  # Enough words left
                and all(
                    [
                        pattern[j] in mylist[i + j]
                        for j in range(i, min(len(pattern), len(mylist[i:])))
                    ]  # All words in the pattern match
                )
                and (
                    tags is None
                    or all(  # All tags are O
                        [
                            tags[j] == "O"
                            for j in range(i, min(i + len(pattern), len(tags)))
                        ]
                    )
                )
            ):
                matches.append(i)

    return matches


def subfinder(
    mylist: List[str],
    pattern: List[str],
    tags: Optional[List[str]] = None,
    agglutinative_language: bool = False,
) -> List[int]:
    matches: List[int] = find_sublist(
        mylist=mylist,
        pattern=pattern,
        tags=tags,
        agglutinative_language=agglutinative_language,
    )

    if len(matches) == 0:
        # Lower everything and remove punctuation
        mylist = [
            x.lower().translate(str.maketrans("", "", string.punctuation))
            for x in mylist
        ]
        # mylist = [x for x in mylist if x]
        pattern = [
            x.lower().translate(str.maketrans("", "", string.punctuation))
            for x in pattern
        ]
        # pattern = [x for x in pattern if x]

        matches: List[int] = find_sublist(
            mylist=mylist,
            pattern=pattern,
            tags=tags,
            agglutinative_language=agglutinative_language,
        )

    return matches


def count_predictions(
    predictions: List[str],
    target_words: List[str],
    task_labels: List[str],
    labels_have_punctuation: bool,
    top_k: int = None,
    agglutinative_language: bool = False,
) -> Dict[str, Dict[str, int]]:
    if top_k is not None:
        predictions = predictions[:top_k]

    prediction_counter: Dict[str, Dict[str, int]] = {}
    """
       {
        Person: {Obama: 2,Biden:1}
        Location: {USA: 1,UK: 1}
        }
       """
    if not task_labels:
        return prediction_counter
    for prediction in predictions:
        pred_labels, pred_label_types = get_labels(
            prediction, task_labels, labels_have_punctuation=labels_have_punctuation
        )
        # print(f"Prediction: {prediction}")
        # print(f"Pred labels: {pred_labels}")

        for text, label_class in zip(pred_labels, pred_label_types):
            if task_labels is None or label_class in task_labels:
                # print(f"Subfinder: mylist: {target_words}, pattern: {text.split()}")
                matches = subfinder(
                    mylist=target_words,
                    pattern=text.split(),
                    tags=None,
                    agglutinative_language=agglutinative_language,
                )
                # print(f"text: {text}, label_class: {label_class}, matches: {matches}")

                if len(matches) > 0:
                    if label_class not in prediction_counter:
                        prediction_counter[label_class] = {}
                    if text not in prediction_counter[label_class]:
                        prediction_counter[label_class][text] = 0
                    prediction_counter[label_class][text] += 1

    return prediction_counter


def get_sentence(
    file: TextIO, set_unique_label: bool = False
) -> (List[str], List[str], List[str], List[str]):
    words: List[str] = []
    labels: List[str] = []

    line: str = file.readline().rstrip().strip()
    while line:
        # print(line)
        # if line.startswith("-DOCSTART-"):
        #    next(file)
        #    line = file.readline().rstrip().strip()
        #    continue

        word: str
        label: str
        try:
            word, label = line.split()
        except ValueError:
            try:
                word, label, _ = line.split()
            except ValueError:
                raise ValueError(f"Error splitting line: {line}")

        words.append(word)
        labels.append(label)

        line = file.readline().rstrip().strip()

    labels = rewrite_labels(labels, encoding="iob2")

    labelled_entities: List[str] = []
    labelled_entities_labels: List[str] = []
    current_label: List[str] = []

    if set_unique_label:
        new_labels = []
        for label in labels:
            if label != "O":
                new_labels.append(f"{label[:1]}-TARGET")
            else:
                new_labels.append(label)
        labels = new_labels

    for word, label in zip(words, labels):
        if label.startswith("B-") or label.startswith("U-"):
            if current_label:
                labelled_entities.append(" ".join(current_label))

            current_label = [word]
            labelled_entities_labels.append(f"{label[2:]}")

        elif label.startswith("I-") or label.startswith("L-"):
            current_label.append(word)
        else:
            if current_label:
                labelled_entities.append(" ".join(current_label))
                current_label = []

    if current_label:
        labelled_entities.append(" ".join(current_label))

    assert len(words) == len(labels), (
        f"Error redding sentence. "
        f"len(words)={len(words)}, "
        f"len(labels)={len(labels)}. "
        f"words: {words}, "
        f"labels: {labels}"
    )
    assert len(labelled_entities) == len(labelled_entities_labels), (
        f"Error redding sentence. "
        f"len(labelled_entities)={len(labelled_entities)}, "
        f"len(labelled_entities_labels)={len(labelled_entities_labels)}.\n"
        f"words: {words}\n"
        f"labels: {labels}\n"
        f"labelled_entities: {labelled_entities}\n"
        f"labelled_entities_labels: {labelled_entities_labels}.\n"
        f"file: {file.name}"
    )

    return words, labels, labelled_entities, labelled_entities_labels


def read_all_sentences_tsv(
    dataset_path: str, set_unique_label: bool = False
) -> (List[List[str]], List[List[str]], List[List[str]], List[List[str]]):
    # print(f"Reading dataset from {dataset_path}.")
    sentences_words: List[List[str]] = []
    sentences_labels: List[List[str]] = []
    sentences_labelled_entities: List[List[str]] = []
    sentences_labelled_entities_labels: List[List[str]] = []

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        words, labels, labelled_entities, labelled_entities_labels = get_sentence(
            file=dataset_file,
            set_unique_label=set_unique_label,
        )
        while words:
            sentences_words.append(words)
            sentences_labels.append(labels)
            sentences_labelled_entities.append(labelled_entities)
            sentences_labelled_entities_labels.append(labelled_entities_labels)
            words, labels, labelled_entities, labelled_entities_labels = get_sentence(
                file=dataset_file,
                set_unique_label=set_unique_label,
            )

    # print(f"Read {len(sentences_words)} sentences from {dataset_path}.")

    return (
        sentences_words,
        sentences_labels,
        sentences_labelled_entities,
        sentences_labelled_entities_labels,
    )


def format_sentence_with_labels(
    words: List[str],
    labelled_entities: List[str],
    labelled_entities_labels: List[str],
) -> (str, str):
    """
    Obama   B-PER
    went    O
    to      O
    New     B-LOC
    York    I-LOC
    .       O

    Source: Obama went to New York . <Person>[None]</Person><Location>[None]</Location>
    Target: <Person>Obama</Person><Location>New York</Location>
    """

    source = []
    target = []

    for entity, label in zip(labelled_entities, labelled_entities_labels):
        try:
            label_name = label2name(label)
        except KeyError:
            raise KeyError(f"Unknown label {label}.")
        source.append(f"<{label_name}>[None]</{label_name}>")
        target.append(f"<{label_name}>{entity}</{label_name}>")

    return " ".join(words) + " " + "".join(source), "".join(target)


def get_task_labels(jsonl_path: str) -> List[str]:
    task_labels = set()
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            json_line = json.loads(line)
            source_label = json_line["source_label"]
            if type(source_label) is not list:
                source_label = source_label.split()
            for label in source_label:
                if label != "O":
                    task_labels.add(label[2:])

    return list(set(task_labels))


def check_labels_have_punctuation(jsonl_path: str) -> bool:
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            json_line = json.loads(line)
            source_entity = json_line["source_entity"]
            for word in source_entity.split():
                if word in string.punctuation:
                    print(
                        f"IMPORTANT! We found that the labels contain words than are punctuation. "
                        f"We won't clean the punctuation from the predictions. "
                    )
                    print(f'For example "{word}" in "{source_entity}"')
                    return True
    print(
        f"IMPORTANT! We found that the labels DO NOT contain words than are punctuation. "
        f"We we will clean the punctuation from the predictions. "
    )
    return False


def get_tags(file_name: str) -> (List[str], List[str]):
    _, _, _, labels = read_all_sentences_tsv(file_name, set_unique_label=False)
    # list of list to lists
    labels = set([item for sublist in labels for item in sublist])
    labels = [label2name(label) for label in labels]
    start_tags = [f"<{label}>" for label in labels]
    end_tags = [f"</{label}>" for label in labels]
    return start_tags, end_tags
