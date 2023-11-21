# T-Projection Examples

The examples correspond to the experiments performed in our  Findings of the EMNLP 2023 paper: [T-Projection: High Quality Annotation Projection for Sequence Labeling Tasks](https://arxiv.org/abs/2212.10548). 

You can get all the required datasets by running the [download_datasets.py](../datasets/download_datasets.py) script.  
Please, check the [datasets/README.md](../datasets/README.md) file for more information about the datasets and the corresponding citations. 

```bash
cd datasets/
python3 download_datasets.py
```



We provide the following examples:
- Test_t-projection.sh: A small test to check that everything is working properly. **This test does not use optimal hyperparameters**. 
- named_entity_recognition_Europarl.sh: Annotation projection from English to Spanish, German and Italian. We use the CoNLL03 data for training. 
- named_entity_recognition_MasakhaNER.sh: Annotation projection from English into low-resource African languages. We use the CoNLL03 data for training.
- opinion_target_extraction_ABSA.sh: Annotation projection from English to Spanish, French, Russian and Turkish. We use the SemEval 2016 data for training.
- argument_mining_abstRCT.sh: Annotation projection from English to Spanish. We use the AbstRCT Neoplasm data for training. 

<p align="center">
    <br>
    <img src="../images/Tasks.jpg">
</p>