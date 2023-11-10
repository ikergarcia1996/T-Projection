# Datasets

This directory contains the datasets used in the paper. Run the following command to download the datasets:

```bash
python3 download_datasets.py
```

The script will download the datasets and extract them in the `dataset/data` directory. **Please don't forger to cite the 
original papers if you use any of the datasets**. The datasets are:

## CoNLL-2003
Task: Named Entity Recognition (NER)  
Source: https://www.clips.uantwerpen.be/conll2003/ner/  
Download source: https://huggingface.co/datasets/conll2003  
Paper: https://aclanthology.org/W03-0419/  
Citation:
```
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://aclanthology.org/W03-0419",
    pages = "142--147",
}
```

## EuroParl-NER
Task: Named Entity Recognition (NER)
Source: https://github.com/ixa-ehu/ner-evaluation-corpus-europarl  
Download source: https://github.com/ixa-ehu/ner-evaluation-corpus-europarl  
Paper: https://aclanthology.org/L18-1557/  
Citation:
```
@inproceedings{agerri-etal-2018-building,
    title = "Building Named Entity Recognition Taggers via Parallel Corpora",
    author = "Agerri, Rodrigo  and
      Chung, Yiling  and
      Aldabe, Itziar  and
      Aranberri, Nora  and
      Labaka, Gorka  and
      Rigau, German",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Hasida, Koiti  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios  and
      Tokunaga, Takenobu",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1557",
}
```

## MasakhaNer2
Task: Named Entity Recognition (NER)  
Source: https://github.com/masakhane-io/masakhane-ner  
Download source: https://huggingface.co/datasets/masakhane/masakhaner2  
Paper: https://aclanthology.org/2022.emnlp-main.298/  
Citation:  
```
@inproceedings{adelani-etal-2022-masakhaner,
    title = "{M}asakha{NER} 2.0: {A}frica-centric Transfer Learning for Named Entity Recognition",
    author = "Adelani, David  and
      Neubig, Graham  and
      Ruder, Sebastian  and
      Rijhwani, Shruti  and
      Beukman, Michael  and
      Palen-Michel, Chester  and
      Lignos, Constantine  and
      Alabi, Jesujoba  and
      Muhammad, Shamsuddeen  and
      Nabende, Peter  and
      Dione, Cheikh M. Bamba  and
      Bukula, Andiswa  and
      Mabuya, Rooweither  and
      Dossou, Bonaventure F. P.  and
      Sibanda, Blessing  and
      Buzaaba, Happy  and
      Mukiibi, Jonathan  and
      Kalipe, Godson  and
      Mbaye, Derguene  and
      Taylor, Amelia  and
      Kabore, Fatoumata  and
      Emezue, Chris Chinenye  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Gitau, Catherine  and
      Munkoh-Buabeng, Edwin  and
      Memdjokam Koagne, Victoire  and
      Tapo, Allahsera Auguste  and
      Macucwa, Tebogo  and
      Marivate, Vukosi  and
      Elvis, Mboning Tchiaze  and
      Gwadabe, Tajuddeen  and
      Adewumi, Tosin  and
      Ahia, Orevaoghene  and
      Nakatumba-Nabende, Joyce  and
      Mokono, Neo Lerato  and
      Ezeani, Ignatius  and
      Chukwuneke, Chiamaka  and
      Oluwaseun Adeyemi, Mofetoluwa  and
      Hacheme, Gilles Quentin  and
      Abdulmumin, Idris  and
      Ogundepo, Odunayo  and
      Yousuf, Oreen  and
      Moteu, Tatiana  and
      Klakow, Dietrich",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.298",
    doi = "10.18653/v1/2022.emnlp-main.298",
    pages = "4488--4508",
    abstract = "African languages are spoken by over a billion people, but they are under-represented in NLP research and development. Multiple challenges exist, including the limited availability of annotated training and evaluation datasets as well as the lack of understanding of which settings, languages, and recently proposed methods like cross-lingual transfer will be effective. In this paper, we aim to move towards solutions for these challenges, focusing on the task of named entity recognition (NER). We present the creation of the largest to-date human-annotated NER dataset for 20 African languages. We study the behaviour of state-of-the-art cross-lingual transfer methods in an Africa-centric setting, empirically demonstrating that the choice of source transfer language significantly affects performance. While much previous work defaults to using English as the source language, our results show that choosing the best transfer language improves zero-shot F1 scores by an average of 14{\%} over 20 languages as compared to using English.",
}
```

## ABSA SemEval 2016
Task: Opinion Target Extraction (OTE)  
English dataset source: https://alt.qcri.org/semeval2016/task5/  
Paper: https://aclanthology.org/S16-1002/  
Spanish, French, Russian, Turkish source: https://github.com/ikergarcia1996/Easy-Label-Projection  
Paper: https://aclanthology.org/2022.findings-emnlp.478/  
Download source: https://huggingface.co/datasets/HiTZ/Multilingual-Opinion-Target-Extraction  
Citation:  
```
@inproceedings{pontiki-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 5: Aspect Based Sentiment Analysis",
    author = {Pontiki, Maria  and
      Galanis, Dimitris  and
      Papageorgiou, Haris  and
      Androutsopoulos, Ion  and
      Manandhar, Suresh  and
      AL-Smadi, Mohammad  and
      Al-Ayyoub, Mahmoud  and
      Zhao, Yanyan  and
      Qin, Bing  and
      De Clercq, Orph{\'e}e  and
      Hoste, V{\'e}ronique  and
      Apidianaki, Marianna  and
      Tannier, Xavier  and
      Loukachevitch, Natalia  and
      Kotelnikov, Evgeniy  and
      Bel, Nuria  and
      Jim{\'e}nez-Zafra, Salud Mar{\'\i}a  and
      Eryi{\u{g}}it, G{\"u}l{\c{s}}en},
    editor = "Bethard, Steven  and
      Carpuat, Marine  and
      Cer, Daniel  and
      Jurgens, David  and
      Nakov, Preslav  and
      Zesch, Torsten",
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S16-1002",
    doi = "10.18653/v1/S16-1002",
    pages = "19--30",
}

@inproceedings{garcia-ferrero-etal-2022-model,
    title = "Model and Data Transfer for Cross-Lingual Sequence Labelling in Zero-Resource Settings",
    author = "Garc{\'\i}a-Ferrero, Iker  and
      Agerri, Rodrigo  and
      Rigau, German",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.478",
    doi = "10.18653/v1/2022.findings-emnlp.478",
    pages = "6403--6416",
    abstract = "Zero-resource cross-lingual transfer approaches aim to apply supervised modelsfrom a source language to unlabelled target languages. In this paper we performan in-depth study of the two main techniques employed so far for cross-lingualzero-resource sequence labelling, based either on data or model transfer. Although previous research has proposed translation and annotation projection(data-based cross-lingual transfer) as an effective technique for cross-lingualsequence labelling, in this paper we experimentally demonstrate that highcapacity multilingual language models applied in a zero-shot (model-basedcross-lingual transfer) setting consistently outperform data-basedcross-lingual transfer approaches. A detailed analysis of our results suggeststhat this might be due to important differences in language use. Morespecifically, machine translation often generates a textual signal which isdifferent to what the models are exposed to when using gold standard data,which affects both the fine-tuning and evaluation processes. Our results alsoindicate that data-based cross-lingual transfer approaches remain a competitiveoption when high-capacity multilingual language models are not available.",
}
```

## AbstRCT 
Task: Argument Component Detection
English Dataset Source: https://gitlab.com/tomaye/abstrct   
Paper: https://ecai2020.eu/papers/1470_paper.pdf   
Spanish Dataset Source: https://github.com/ragerri/abstrct-projections  
Paper: https://arxiv.org/abs/2301.10527  
Download source: https://github.com/ragerri/abstrct-projections  
Citation:   
```
@inproceedings{mayer2020ecai,
  author    = {Tobias Mayer and
               Elena Cabrio and
               Serena Villata},
  title     = {Transformer-Based Argument Mining for Healthcare Applications},
  booktitle = {{ECAI} 2020 - 24th European Conference on Artificial Intelligence},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {325},
  pages     = {2108--2115},
  publisher = {{IOS} Press},
  year      = {2020},
}

@article{yeginbergenova2023cross,
  title={Cross-lingual Argument Mining in the Medical Domain},
  author={Yeginbergenova, Anar and Agerri, Rodrigo},
  journal={arXiv preprint arXiv:2301.10527},
  year={2023}
}

```
