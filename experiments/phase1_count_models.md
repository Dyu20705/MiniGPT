# Phase 1 Count Models

## Data

- Corpus path: `data/raw/tiny_corpus.txt`
- Absolute corpus path: `C:\Users\Admin\Src\Cod\ai\MiniGPT\data\raw\tiny_corpus.txt`
- Corpus SHA-256: `11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4`
- Vocabulary size: 47
- Train tokens: 1341
- Validation tokens: 167
- Test tokens: 169
- Split ratios: train 0.8, validation 0.1, test remainder
- Generation seed: 42

## Models

Main experiment uses `alpha = 0.1`. Bigram also uses `backoff_alpha = 0.1`.

| Model | Split | Mean NLL | Perplexity | Predictions |
|---|---:|---:|---:|---:|
| Unigram | Train | 3.189685 | 24.280786 | 1340 |
| Bigram | Train | 1.621591 | 5.061138 | 1340 |
| Unigram | Validation | 3.167593 | 23.750239 | 166 |
| Bigram | Validation | 1.594022 | 4.923512 | 166 |
| Unigram | Test | 3.305356 | 27.258243 | 168 |
| Bigram | Test | 1.750866 | 5.759588 | 168 |

Unigram is evaluated on `ids[2:end]` when compared with bigram, so both models score the same target positions. Splits are evaluated independently, without connecting the last token of one split to the first token of the next split.

## Smoothing Sweep

`backoff_alpha` is fixed at 0.1 for bigram during the sweep. Best alpha is selected only from validation NLL.

| Alpha | Unigram validation NLL | Unigram validation PPL | Bigram validation NLL | Bigram validation PPL | Bigram test PPL | Status | Note |
|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | 3.167542 | 23.749048 | 1.464979 | 4.327454 | Inf | validation finite, test Inf | Best validation score, but unseen test transition causes Inf. |
| 0.01 | 3.167547 | 23.749147 | 1.480102 | 4.393396 | 5.301170 | finite | Small smoothing fixes unseen test transitions. |
| 0.10 | 3.167593 | 23.750239 | 1.594022 | 4.923512 | 5.759588 | finite | Main experiment setting. |
| 1.00 | 3.168806 | 23.779079 | 2.134217 | 8.450429 | 9.787836 | finite | Heavy smoothing weakens observed bigram structure. |

Best unigram alpha by validation NLL: 0.0.

Best bigram alpha by validation NLL: 0.0.

The test set is not used for alpha selection. Alpha 0.0 produces Inf on bigram test perplexity because at least one test transition was unseen in train.

## Frequent Transitions

| Rank | Previous | Next | Count | P(next\|prev) |
|---:|---|---|---:|---:|
| 1 | `" "` | `"t"` | 70 | 0.24281261 |
| 2 | `" "` | `"h"` | 47 | 0.16314513 |
| 3 | `"n"` | `"g"` | 41 | 0.31688512 |
| 4 | `"n"` | `" "` | 36 | 0.27833462 |
| 5 | `"."` | `" "` | 35 | 0.78523490 |
| 6 | `" "` | `"n"` | 30 | 0.10426048 |
| 7 | `" "` | `"m"` | 24 | 0.08347766 |
| 8 | `" "` | `"đ"` | 24 | 0.08347766 |
| 9 | `"g"` | `" "` | 24 | 0.46615087 |
| 10 | `"y"` | `" "` | 24 | 0.69452450 |
| 11 | `" "` | `"c"` | 18 | 0.06269484 |
| 12 | `" "` | `"d"` | 18 | 0.06269484 |
| 13 | `"c"` | `"h"` | 18 | 0.52161383 |
| 14 | `"h"` | `" "` | 18 | 0.17287488 |
| 15 | `"h"` | `"ì"` | 18 | 0.17287488 |
| 16 | `"l"` | `"i"` | 18 | 0.63066202 |
| 17 | `"m"` | `"ô"` | 18 | 0.39606127 |
| 18 | `"n"` | `"h"` | 18 | 0.13955281 |
| 19 | `"t"` | `"r"` | 18 | 0.21120187 |
| 20 | `"ì"` | `"n"` | 18 | 0.79735683 |

## Samples

Unigram sample, 300 tokens:

```text
t i  di  i yhy gstđnônmyo kiôg  gđk  mìhđgti ạđgt y.ôhti  tnh đsyôdáaọờựảàagôhpgjr,okhxsxếđ ìguhtttn    ti lc.đnn.h i e nìoiấ.n em nú  imkpi iạn ảm lể. t o cữ  rcgm ế.nnnie neữopoữ ìtn   lmcì htả.ẹsnh iiìpym .ulểôtc ênhôềì.tựàntàong p ốttờàhpemonôùrupeho  miìccnđctn nihiđinyh j ìghểọ. iệon i ôú ultc
```

Bigram sample, prompt plus 300 new tokens:

```text
xo m c m hìnhìn juúảnhôiểm máy h ký h nhìxiểm tảnh tesmôi hếp.
ếpiệnhuẹờựýìnay thìn sthểmôrờiệnhìngoke c th n họchne m jtrờinàoke n nhdáng. m môi tra.te t smhố
mô ki đangin.
xiớiếpốmômô heliay kýế đẹp.
xiệuất hếpýýo.
únesếteoờêngômô gi.ảtrờìng ng. trốn. ra ngô húấnhúng máùngô ch thìảốn mômô m tả tiế 
```

## Qualitative Observations

Unigram learns global character frequency. Its sample has many frequent characters and spaces, but local spelling and punctuation structure are weak because every token is sampled independently.

Bigram learns one-token local structure. It improves short transitions such as spaces before common letters, Vietnamese character pairs, period followed by space, and repeated local fragments. This is reflected by much lower validation and test perplexity at alpha 0.1.

The bigram sample has more word-like local chunks and line/punctuation behavior than unigram, but it still repeats short fragments and cannot maintain longer syntax because its context is only the previous token.

Smoothing matters mostly for sparse transitions. Alpha 0.0 gives the best validation NLL in this split, but it produces Inf on bigram test perplexity due to unseen transitions. Small alpha values keep most observed structure while avoiding Inf on held-out data.

## Conclusion

The bigram count model is a strong improvement over the unigram baseline on this corpus for the main alpha 0.1 setting: validation perplexity drops from 23.750239 to 4.923512, and test perplexity drops from 27.258243 to 5.759588. The alpha sweep shows that unsmoothed bigram is attractive on validation but unsafe for held-out generalization because unseen test transitions can make perplexity infinite.
