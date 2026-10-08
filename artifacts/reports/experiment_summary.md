# Experiment summary

## 구현 요약

NumPy Raw/Log/Double TF-IDF, direct Okapi BM25 and sparse term postings. Existing SVM/loader/redaction preserved.

## 검증 및 실험 결과

Dataset: 20 Newsgroups, 18,309 documents, 20 categories.
Train/test: 14,647/3,662, stratified 80/20; seed 42.
Quality: 100 held-out document queries, k=10. Latency: warm-up + 5 repeats.
SVM: hinge, epochs=6, batch_size=128, train-only fit.
Runtime load (excluded from query latency): 4837.233 ms.

| TF | Accuracy | Macro F1 | P@10 | MAP@10 |
|---|---:|---:|---:|---:|
| raw | 0.767067 | 0.754385 | 0.521000 | 0.434458 |
| log | 0.770071 | 0.757423 | 0.557000 | 0.478212 |
| double | 0.768979 | 0.753667 | 0.528000 | 0.456887 |

| Method (train corpus) | P@10 | MAP@10 | Mean ms | P95 ms |
|---|---:|---:|---:|---:|
| tfidf | 0.521000 | 0.434458 | 12.546 | 31.254 |
| bm25 | 0.570000 | 0.490718 | 12.546 | 31.088 |

Speed uses short presentation queries, separate from held-out quality queries.

| Documents | Scan mean ms | Inverted mean ms | Scan P95 ms | Inverted P95 ms | Speedup | Extra posting MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 9.896 | 0.308 | 10.206 | 0.415 | 32.09x | 0.774 |
| 5000 | 48.501 | 0.488 | 49.639 | 0.604 | 99.44x | 4.068 |
| 10000 | 97.403 | 0.712 | 99.111 | 0.970 | 136.90x | 8.027 |
| 18309 | 175.790 | 1.133 | 178.284 | 1.592 | 155.10x | 14.427 |

### Actual per-query differences (held-out quality)

AP@10: BM25 better on 55 queries; TF-IDF better on 32; tied on 13.

| Query document ID | TF-IDF P@10 | BM25 P@10 | TF-IDF AP@10 | BM25 AP@10 |
|---:|---:|---:|---:|---:|
| 1264 | 0.5000 | 1.0000 | 0.3267 | 1.0000 |
| 18693 | 0.2000 | 0.8000 | 0.1250 | 0.7889 |
| 10583 | 0.4000 | 0.8000 | 0.2275 | 0.7800 |
| 18749 | 0.9000 | 0.5000 | 0.8354 | 0.3796 |
| 7326 | 1.0000 | 0.7000 | 1.0000 | 0.6115 |
| 5678 | 0.7000 | 0.2000 | 0.5163 | 0.1400 |

Numerical validation:
- raw: PASS, maximum absolute error=5.440e-15; {'smooth_idf': True, 'sublinear_tf': False, 'norm': 'l2', 'use_idf': True, 'dtype': 'float64'}.
- log: PASS, maximum absolute error=3.955e-16; {'smooth_idf': True, 'sublinear_tf': True, 'norm': 'l2', 'use_idf': True, 'dtype': 'float64'}.
- double: PASS, maximum absolute error=0.000e+00; hand calculation.
- bm25: PASS, maximum absolute error=1.110e-16; hand calculation.

Stopwords paired experiment (one seed, separate from historical ten-seed report):

| Condition | Accuracy | Macro F1 | P@10 | MAP@10 |
|---|---:|---:|---:|---:|
| default_stop_words | 0.767067 | 0.754385 | 0.521000 | 0.434458 |
| no_stop_words | 0.762425 | 0.752118 | 0.465000 | 0.386463 |

### Top-5 (full demo corpus)

BM25 and cosine scores have different scales; only ranks/labels and token lengths are compared.

Query: `space shuttle orbit`; common documents: 1.

| Rank | TF-IDF ID / tokens / category | BM25 ID / tokens / category |
|---:|---|---|
| 1 | 4672 / 689 / sci.space | 3190 / 181 / sci.space |
| 2 | 4389 / 1074 / sci.space | 12906 / 96 / sci.space |
| 3 | 5788 / 230 / sci.space | 3319 / 337 / sci.space |
| 4 | 12888 / 145 / sci.space | 4672 / 689 / sci.space |
| 5 | 10571 / 991 / sci.space | 12548 / 70 / sci.space |

Query: `baseball pitcher season`; common documents: 3.

| Rank | TF-IDF ID / tokens / category | BM25 ID / tokens / category |
|---:|---|---|
| 1 | 6874 / 2 / rec.sport.baseball | 5366 / 107 / rec.sport.baseball |
| 2 | 5366 / 107 / rec.sport.baseball | 12163 / 18 / rec.sport.baseball |
| 3 | 14800 / 99 / rec.sport.baseball | 4283 / 53 / rec.sport.baseball |
| 4 | 4283 / 53 / rec.sport.baseball | 14800 / 99 / rec.sport.baseball |
| 5 | 15809 / 82 / rec.sport.baseball | 13018 / 58 / rec.sport.baseball |

Query: `computer graphics image`; common documents: 1.

| Rank | TF-IDF ID / tokens / category | BM25 ID / tokens / category |
|---:|---|---|
| 1 | 8050 / 52 / comp.graphics | 8050 / 52 / comp.graphics |
| 2 | 18102 / 4317 / comp.graphics | 14852 / 53 / comp.graphics |
| 3 | 11244 / 4284 / comp.graphics | 13298 / 71 / comp.graphics |
| 4 | 12488 / 142 / comp.graphics | 3752 / 97 / comp.graphics |
| 5 | 10085 / 3697 / comp.graphics | 9748 / 122 / comp.graphics |

### Misclassification examples

raw confusion matrix: `tf_raw_confusion_matrix.png`; per-class precision/recall/F1: `tf_raw_per_class.csv`.
- ID 10167: alt.atheism → rec.autos: poor matthew a million posters to call you car drivers and he chooses me a non car owner
- ID 17783: sci.med → rec.sport.baseball: report them to your local bbb better business bureau
- ID 2608: comp.os.ms-windows.misc → comp.graphics: this should be in the faq it only get asked about twenty times a month there is no that s right no telnet access to compuserve no ftp access no gopher access no internet services except e mail you can telnet to hermes merit edu but that
- ID 3292: talk.politics.misc → talk.politics.guns: the paranoid assertion that the batf fired first in an unprovoked assault assumes that the batf were on a death wish had they expected the b d to be anything other than peacefull citizens who would accept a search authorized by a court they
- ID 8072: comp.graphics → comp.sys.ibm.pc.hardware: the does exist it s number was licensed to philips to make their own variant this chip includes extra featurfes such as more i o ports i c bus making it more microcontroller like
log confusion matrix: `tf_log_confusion_matrix.png`; per-class precision/recall/F1: `tf_log_per_class.csv`.
- ID 10167: alt.atheism → rec.autos: poor matthew a million posters to call you car drivers and he chooses me a non car owner
- ID 17783: sci.med → comp.graphics: report them to your local bbb better business bureau
- ID 2608: comp.os.ms-windows.misc → comp.graphics: this should be in the faq it only get asked about twenty times a month there is no that s right no telnet access to compuserve no ftp access no gopher access no internet services except e mail you can telnet to hermes merit edu but that
- ID 3292: talk.politics.misc → talk.politics.guns: the paranoid assertion that the batf fired first in an unprovoked assault assumes that the batf were on a death wish had they expected the b d to be anything other than peacefull citizens who would accept a search authorized by a court they
- ID 8072: comp.graphics → comp.sys.ibm.pc.hardware: the does exist it s number was licensed to philips to make their own variant this chip includes extra featurfes such as more i o ports i c bus making it more microcontroller like
double confusion matrix: `tf_double_confusion_matrix.png`; per-class precision/recall/F1: `tf_double_per_class.csv`.
- ID 10167: alt.atheism → rec.autos: poor matthew a million posters to call you car drivers and he chooses me a non car owner
- ID 17783: sci.med → rec.sport.baseball: report them to your local bbb better business bureau
- ID 2608: comp.os.ms-windows.misc → comp.graphics: this should be in the faq it only get asked about twenty times a month there is no that s right no telnet access to compuserve no ftp access no gopher access no internet services except e mail you can telnet to hermes merit edu but that
- ID 3292: talk.politics.misc → talk.politics.guns: the paranoid assertion that the batf fired first in an unprovoked assault assumes that the batf were on a death wish had they expected the b d to be anything other than peacefull citizens who would accept a search authorized by a court they
- ID 8072: comp.graphics → comp.sys.ibm.pc.hardware: the does exist it s number was licensed to philips to make their own variant this chip includes extra featurfes such as more i o ports i c bus making it more microcontroller like

## 한계와 개선 방향

Same-category relevance is a proxy, not a semantic judgment. The recorded query subset limits generalization; this is one split and one timing run. Empty/OOV quality queries stay in the denominator; zero-score ranks fill by ID to maintain fixed-k metric definitions. Timing excludes load and compares identical queries per pair. Array memory excludes Python overhead and transient allocations.

BoW discards word order and cannot resolve polysemy/context. Tokenization, stopwords and existing redaction lose information. Raw TF overweights repetition; log limits it; double stresses presence. BM25 saturates TF and normalizes length, which may help concise documents but needs tuning. Postings add construction/memory costs and frequent/long queries still touch many documents. Overlapping topic vocabularies and short/ambiguous documents cause SVM errors. Follow-up: repeated splits, judged queries, parameter studies and linguistic features.

### Environment and reproduction

Executed: 2026-10-08T09:51:42.716835+00:00
Corpus fingerprint: `69168bb0a2f808f1278ae3bb9050aaa5537d9e36607352a2d0b8b3e8fc50b396`

```json
{
  "python": "3.13.7 (main, Mar  3 2026, 12:19:54) [GCC 15.2.0]",
  "platform": "Linux-6.17.8-orbstack-00308-g8f9c941121b1-x86_64-with-glibc2.42",
  "processor": "",
  "packages": {
    "numpy": "2.5.2",
    "scikit-learn": "1.9.0",
    "matplotlib": "3.11.1"
  },
  "machine": "x86_64",
  "cpu_count": 6,
  "thread_environment": {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": null
  }
}
```

Run `python main.py benchmark --queries 100 --repeats 5 --epochs 6 --batch-size 128` after build. Full configuration and split/query IDs are in retrieval_benchmark.json.
