# NER_PROJECT

The project paper can be viewed [Here](Meta_Learning_Project-5.pdf)

## Deep Learning for data extraction in scientific literature

### Purpose

In this study, we will examine the performance of two different models in the context of Few-shot learning. We will be looking at ProtoBert [ProtoBert](https://aclanthology.org/2022.acl-long.521) model, an extension based on prototypical networks supported on a BERT encoder [[1]](#1). Additionally, we will be comparing the performance of this latter with a more complicated model based on Meta-learning. Specifically, the Decomposed Meta-Learning (DML) algorithm presented by Ma et al. [[2]](#2). This much more complex approach sequentially addresses few-shot span detection and few-shot entity typing with two decomposed models. They then combine the two to make one predictor and perform NER. 

In other words we aim to benchmark the performance of this decomposed meta-learning approach versus a more "simple" and straightforward implementation, like ProtoBert, in the few-shot NER setting. Specifically, we will be evaluating them on the PICO (Population, Intervention, Comparison, Outcome) Corpus, a collection of bio-medical research articles. We will also compare our results with Ma et al. on the Few-NERD [[3]](#3) data set with similar models. The primary objective of our research is to evaluate how well two models can perform an NER task on bio-medical texts in a Few-shot manner.

### Models considered to tackle this project:

1. ProtoBERT
2. Decomposed META-Learning

### Conclusion

Our empirical results highlighted the effectiveness of the ProtoBERT model when tested on the PICO Corpus, showing robust generalization capabilities. However, the DML model performance seemed to be affected by its complexity, which may be a subject for further investigation. Overfitting due to over-specialization to the training data is one possible explanation, revealing an interesting trade-off between model complexity and generalizability. A finer investigation into how model complexity, the number of classes, and overfitting interplay could be a future research direction. Interestingly, we observed that the data quality and the class distribution could be another influencing factor for model performance. Investigating the knowledge correlation among entity types, as it pertains to knowledge transfer in few-shot learning, could be interesting to look at. Refining the class distribution to match the data set characteristics could also potentially improve model performance.

### Relevant articles

- [Corpus for training to extract biomedical literature](https://aclanthology.org/2022.wiesp-1.4.pdf)
- https://ieeexplore.ieee.org/document/9039685
- [Survey on DL models for NER](https://arxiv.org/pdf/1603.01360.pdf)

## References

<a id="1">[1]</a> 
Jacob Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).

<a id="2">[2]</a>
Tingting Ma et al. “Decomposed Meta-Learning for Few-Shot Named Entity Recognition”. In: Findings of the Association for Computational Linguistics.

<a id="3">[3]</a>
Ning Ding et al. “Few-NERD: A Few-shot Named Entity Recognition Dataset”. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers).
