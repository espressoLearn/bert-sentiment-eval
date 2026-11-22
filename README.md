# BERT Sentiment Analyzer and Evaluation

Fine-tuned BERT on the Stanford Sentiment Treebank for binary and 5-class sentiment, then compared bert-base against bert-large with a learning-rate search. bert-large reached 93.5% accuracy on binary sentiment and cut wrong-direction errors on the 5-class task by a third. The more useful outcome was the evaluation: it found error patterns the headline metrics hid. Built for a Penn State graduate NLP course (Summer 2026).

## Approach

1. **Start simple.** Binary sentiment first, since it is balanced and needs no reweighting. Use what it reveals to design the 5-class model.
2. **Let the binary model shape the 5-class design.**
   - The 5-class labels are imbalanced about 2 to 1, so the 5-class model uses a class-weighted loss.
   - Neutral text was the hard part, so checkpoints are picked by macro-F1, which keeps small classes from being drowned out.
   - Labels are ordinal (very negative to very positive), so I added adjacent accuracy: a prediction one notch off counts differently from a polarity flip.
3. **Scale up.** Re-ran both tasks on bert-large with a learning-rate grid, fitted into 6 GB of GPU memory with fp16, dynamic padding, Adafactor and gradient checkpointing.
4. **Instrument the runs.** Step logging every 50 steps, timers for each stage, best-checkpoint selection, early stopping, and SMS alerts on milestones and on any exception, so overnight runs report failures right away.
5. **Explain predictions.** LIME and SHAP at each stage.

## Results

| Model | Binary | 5-class |
|---|---|---|
| bert-base | 90.1% accuracy, 0.905 F1 | 48.3% accuracy, 0.488 macro-F1, 91.3% adjacent accuracy |
| bert-large + LR search | 93.5% accuracy, 0.936 F1 | 53.2% accuracy, 0.532 macro-F1, 94.3% adjacent accuracy |

bert-large cut polarity flips on the 5-class task from 96 to 63.

## What I learned

- **Balanced metrics can hide one-sided failure.** The binary model's class metrics looked even, but 14 of its 15 most confident errors were negative sentences called positive at 99%+ confidence.
- **Pick metrics that fit the labels.** 473 of 569 bert-base 5-class errors were only one notch off. Plain accuracy (48%) made the model look far worse than adjacent accuracy (91%) shows it was.
- **Test behavior on inputs the model never saw.** The binary model was never trained on neutral sentences, and how it spread its confidence over them predicted where the 5-class model would struggle.
- **Weigh cost against quality.** The winning 5-class learning rate (1e-5) took 7.4 hours and beat a 31-minute run by just 0.004 macro-F1. When compute matters, the faster run is the better choice.

## Setup

1. Download the Stanford Sentiment Treebank trees (not included) into `data/raw/sst/`.
2. `pip install -r requirements.txt`
3. SMS alerts are optional. Copy `.env.example` to `.env` and fill in the Twilio settings to turn them on. With them blank, the notebook skips texting.

Interactive SHAP plots were removed from the saved notebook to keep the file small. Rerun those cells to regenerate them.
