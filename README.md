
## What is BERT?

BERT is a method of pre-training language representations, meaning that we train
a general-purpose "language understanding" model on a large text corpus (like
Wikipedia), and then use that model for downstream NLP tasks that we care about
(like question answering). BERT outperforms previous methods because it is the
first *unsupervised*, *deeply bidirectional* system for pre-training NLP.

*Unsupervised* means that BERT was trained using only a plain text corpus, which
is important because an enormous amount of plain text data is publicly available
on the web in many languages.

Pre-trained representations can also either be *context-free* or *contextual*,
and contextual representations can further be *unidirectional* or
*bidirectional*. Context-free models such as
[word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) or
[GloVe](https://nlp.stanford.edu/projects/glove/) generate a single "word
embedding" representation for each word in the vocabulary, so `bank` would have
the same representation in `bank deposit` and `river bank`. Contextual models
instead generate a representation of each word that is based on the other words
in the sentence.

BERT was built upon recent work in pre-training contextual representations —
including [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432),
[Generative Pre-Training](https://blog.openai.com/language-unsupervised/),
[ELMo](https://allennlp.org/elmo), and
[ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
— but crucially these models are all *unidirectional* or *shallowly
bidirectional*. This means that each word is only contextualized using the words
to its left (or right). For example, in the sentence `I made a bank deposit` the
unidirectional representation of `bank` is only based on `I made a` but not
`deposit`. Some previous work does combine the representations from separate
left-context and right-context models, but only in a "shallow" manner. BERT
represents "bank" using both its left and right context — `I made a ... deposit`
— starting from the very bottom of a deep neural network, so it is *deeply
bidirectional*.

BERT uses a simple approach for this: We mask out 15% of the words in the input,
run the entire sequence through a deep bidirectional
[Transformer](https://arxiv.org/abs/1706.03762) encoder, and then predict only
the masked words. For example:

```
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

In order to learn relationships between sentences, we also train on a simple
task which can be generated from any monolingual corpus: Given two sentences `A`
and `B`, is `B` the actual next sentence that comes after `A`, or just a random
sentence from the corpus?

```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
```

```
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```

We then train a large model (12-layer to 24-layer Transformer) on a large corpus
(Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb)) for a long time (1M
update steps), and that's BERT.

Using BERT has two stages: *Pre-training* and *fine-tuning*.

**Pre-training** is fairly expensive (four days on 4 to 16 Cloud TPUs), but is a
one-time procedure for each language (current models are English-only, but
multilingual models will be released in the near future). We are releasing a
number of pre-trained models from the paper which were pre-trained at Google.
Most NLP researchers will never need to pre-train their own model from scratch.

**Fine-tuning** is inexpensive. All of the results in the paper can be
replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU,
starting from the exact same pre-trained model. SQuAD, for example, can be
trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of
91.0%, which is the single system state-of-the-art.

The other important aspect of BERT is that it can be adapted to many types of
NLP tasks very easily. In the paper, we demonstrate state-of-the-art results on
sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level
(e.g., NER), and span-level (e.g., SQuAD) tasks with almost no task-specific
modifications.
