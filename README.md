# Intel Image Classification — CNN Architecture Comparison

A systematic, controlled comparison of CNN architectural patterns — batch normalization,
residual connections, depthwise-separable convolutions, and a from-scratch Xception-style
network — followed by transfer learning with a pretrained Xception-41 backbone, all
evaluated on the same 6-class scene classification task.

The goal wasn't to chase the highest possible accuracy. It was to isolate *why* each
technique helps (or doesn't), on a modest-sized dataset that's smaller and less complex
than what most of these techniques were originally designed for.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
— ~14,000 training images and ~3,000 test images across 6 scene categories: buildings,
forest, glacier, mountain, sea, street. Images were resized to 150×150 for the
from-scratch models and 299×299 for the pretrained-backbone models (matching Xception-41's
native training resolution).

An 80/20 train/validation split was carved out of the training set with a fixed seed, kept
identical across every model for a fair comparison. The test set was held out entirely
until final evaluation.

## Methodology

Every model was trained for 15 epochs with the Adam optimizer, `sparse_categorical_crossentropy`
loss, and a `ModelCheckpoint` callback that saved only the best-val-loss weights — meaning
every reported result reflects the best point in training, not whatever the final epoch
happened to look like. This mattered more than once (see below).

The five from-scratch stages were built **cumulatively** — each one adds a single technique
on top of the previous stage — so results reflect the combined effect up to that point, not
each technique in isolation. Two additional models push further: a deeper, more faithful
Xception-style network with dropout and data augmentation, and transfer learning with a
pretrained Xception-41 backbone (first frozen, then fine-tuned).

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Fine-tuned Xception-41 | **0.910** | 0.367 |
| Pretrained Xception-41 (frozen backbone) | 0.892 | 0.359 |
| Xception-Lite V2 (deep + dropout + augmentation) | 0.870 | 0.388 |
| Baseline CNN | 0.856 | 0.425 |
| + Batch Normalization | 0.854 | 0.430 |
| + Depthwise-Separable Convolutions | 0.840 | 0.466 |
| Xception-Lite V1 (entry/middle/exit, no regularization) | 0.828 | 0.451 |
| + Residual Connections | 0.827 | 0.509 |

Per-model training/validation accuracy and loss curves are in
[`training_curves/`](training_curves/), and architecture diagrams
(generated via `keras.utils.plot_model`) are in
[`architecture_diagrams/`](architecture_diagrams/).

Trained model weights (`.keras` checkpoints) are not included in this repo — several
exceed GitHub's 100MB file limit. [Add a Hugging Face Hub / Drive link here if you upload
them separately.]

## Discussion

**Architectural sophistication alone didn't help — at this dataset scale, it mostly hurt.**
Every from-scratch model that added a "best practice" technique without also increasing
depth or adding regularization ended up at or below the plain baseline. Residual
connections, in particular, consistently underperformed (0.827, the weakest result in the
whole comparison). This lines up with why residual connections were introduced in the
first place: they solve a degradation/vanishing-gradient problem specific to *very deep*
networks (dozens to hundreds of layers). At 4 blocks deep, that problem doesn't really
exist yet — so the extra shortcut-path parameters mostly just gave the model more capacity
to overfit with, evidenced by a wider train/validation accuracy gap than the baseline
showed. Depthwise-separable convolutions partially recovered some of that lost ground
(0.840), consistent with their actual selling point: parameter *efficiency*, not
representational power.

**A subtle batch normalization bug shaped a lot of this project, and is worth documenting
explicitly.** Early runs of the batchnorm-augmented models showed wildly unstable
validation metrics — validation loss spiking as high as 3× training loss between adjacent
epochs, with no clear improving trend. The cause was BatchNormalization's default
`momentum=0.99`: at that setting, the running mean/variance estimate used at evaluation
time updates far too slowly to converge within a normal-length training run, so validation
metrics were computed against a stale, inaccurate normalization statistic. Lowering
`momentum` to `0.9` (a 10% adaptation rate per step instead of 1%) fixed this cleanly —
validation curves became smooth and well-behaved across every subsequent model. This is a
good example of a hyperparameter that's easy to overlook (it's rarely touched in
introductory material) but has an outsized effect on training stability specifically for
smaller-to-medium datasets like this one.

**Going deep enough finally let complexity pay off — but only once regularization came
with it.** The first attempt at a full Xception-style network (entry/middle/exit flow, no
dropout, no augmentation) reached 97% training accuracy while validation accuracy stalled
around 82-86% — the clearest overfitting signature in the whole project. The
`ModelCheckpoint` callback caught this and saved an early, still-reasonable checkpoint
(0.828 test accuracy) rather than the badly overfit final weights, but the underlying
problem was clear: this network finally had *enough capacity* to fit the task well, but
nothing was constraining how that capacity got used. Adding dropout and data augmentation
in a second, deeper version (8 middle-flow blocks instead of 4, proper pre-activation
ordering) resolved that — training and validation accuracy tracked much more closely, and
this was the first from-scratch model to clearly and consistently beat the baseline
(0.870). The lesson: depth and regularization are a package deal, not independent levers.

**Transfer learning was the single biggest driver of performance in the entire project.**
Even with the backbone completely frozen — only a small classification head trained from
scratch — the pretrained Xception-41 model (0.892) outperformed every from-scratch
architecture, including the deep, well-regularized one. This is a meaningful result: it
suggests the ~85-87% ceiling the from-scratch models kept converging to wasn't primarily a
capacity or architecture problem — it was a *data* problem. ~14,000 training images spread
across 6 classes just isn't enough for a randomly-initialized network to learn features as
good as what ImageNet's ~1.4 million images already baked into this backbone. Fine-tuning
the backbone's later layers (with BatchNorm layers deliberately kept frozen, and a 100×
lower learning rate to avoid destroying the pretrained weights) pushed this further to
0.910 — the best result of the project — confirming that adapting general-purpose features
to this specific task, rather than relying on frozen general features alone, extracts real
additional value once the foundation is already strong.

**Fine-tuning converged almost immediately — and then quietly overfit for 12 more epochs.**
Looking at the fine-tuning run's loss curve, validation loss reaches its lowest point
around epoch 2-3, while training loss keeps collapsing toward zero all the way through
epoch 15. `ModelCheckpoint` correctly saved that early best point rather than the final
weights, which is exactly why the reported result (0.910) is trustworthy — but it also
means roughly 12 of the 15 fine-tuning epochs were pure wasted compute, actively pushing
the model toward overfitting rather than improving it. In hindsight, 3-5 epochs would
likely have reached the same result in a fraction of the training time, which is worth
knowing given fine-tuning was the slowest stage in the whole project per epoch.

**Overall takeaway:** on a dataset this size, individual architectural techniques from the
CNN literature (batchnorm, residuals, separable convolutions) don't automatically improve
results just because they're "best practices" — several of them measurably hurt when
applied without enough depth or countervailing regularization. What actually moved the
needle was either (a) combining sufficient depth *with* regularization, or (b) sidestepping
the data-scarcity problem entirely via transfer learning. The project's main lesson isn't
about any one technique — it's that architectural choices need to be evaluated relative to
dataset scale, not applied by default.

## Limitations & Future Work

- All comparisons used a fixed 15-epoch budget and constant learning rate for the
  from-scratch stages, to keep the comparison controlled. It's possible some
  underperforming architectures (residual connections in particular) would close the gap
  given more epochs, a learning rate schedule, or a larger dataset.
- The five from-scratch stages train at 150×150 resolution; the transfer learning stages
  use 299×299 (matching the pretrained backbone's native resolution). This is a deliberate,
  documented choice, but it means the transfer learning comparison isn't perfectly
  apples-to-apples with the from-scratch stages.
- Fine-tuning unfroze the entire backbone (aside from BatchNorm layers). Unfreezing only
  the last few blocks is a common alternative that may reduce overfitting risk further and
  would be worth comparing.
- A PyTorch reimplementation of the best-performing architecture would be a natural
  extension, both as a learning exercise and to sanity-check results across frameworks.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Intel_Image_Classification_Final.ipynb
├── architecture_diagrams/   # keras.utils.plot_model() outputs, one per stage
└── training_curves/         # accuracy/loss plots, one per stage

# model_weights/ exists locally but is excluded via .gitignore -- see note above
```

## Running This Project

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Open `Intel_Image_Classification_Final.ipynb` in Jupyter or Google Colab (a T4 GPU
   or better is strongly recommended — several stages, especially the transfer learning
   models, are impractically slow on CPU).
3. Run the Kaggle authentication cell and follow the interactive login prompt (see the
   note in the notebook — no API key should ever be hardcoded).
4. Run all cells top to bottom. Each stage trains, plots its curves, and evaluates on the
   test set automatically.
