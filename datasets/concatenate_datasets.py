from typing import List
import argparse


def concatenate_datasets(
    datasets: List[str], output_file: str, new_line_between_datasets: bool = False
) -> None:
    """
    Concatenate multiple datasets into a single dataset.
    """
    with open(output_file, "w", encoding="utf8") as output:
        for dataset in datasets:
            with open(dataset, "r", encoding="utf8") as f:
                lines = f.readlines()
                lines = [line.strip().rstrip() for line in lines]
                if new_line_between_datasets:
                    if lines[-1] != "":
                        lines.append("")
                else:
                    if lines[-1] == "":
                        lines.pop()
                print("\n".join(lines), file=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", type=str, help="Datasets to concatenate."
    )
    parser.add_argument("-o", "--output", type=str, help="Output file.")
    parser.add_argument(
        "-nl",
        "--new-line-between-datasets",
        action="store_true",
        help="Add a new line between datasets.",
    )

    args = parser.parse_args()
    concatenate_datasets(args.datasets, args.output, args.new_line_between_datasets)
