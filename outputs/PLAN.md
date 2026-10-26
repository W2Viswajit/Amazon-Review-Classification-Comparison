# Amazon Review Classification — implementation plan

Status: planning only; no models have been trained and no results claimed.

## Scope
Compare TF-IDF + logistic regression, genuinely fine-tuned BERT, and a prompted instruction-following LLM on Amazon positive/negative reviews. Only the overview was provided, so exact marking requirements remain unverified.

## Data
Use https://huggingface.co/datasets/fancyzhx/amazon_polarity (public, already labeled). Join title and content. Labels: 0 negative, 1 positive. Stream a reproducibly shuffled subset rather than download the entire corpus. Record sampling seed and source; remove exact text duplicates and cross-split overlap. A buffered streaming shuffle is an approximate sample, not a uniform sample of the entire corpus.

Target: 2,000 balanced training examples and 400 balanced validation examples from the official training split; 200 balanced test examples from the official test split. Preserve the same test IDs for all three methods. Adjust training size only if a measured runtime estimate requires it, and document the change. Do not remove negation or aggressively clean the language.

## Implementation sequence
1. Create a project environment and check PyTorch GPU availability. Python and VS Code are present. Installed PyTorch is 2.7.0+cpu and reports CUDA unavailable; this environment currently cannot use CUDA. Transformers and scikit-learn are installed; datasets and Jupyter are missing. This does not establish whether compatible physical GPU hardware exists.
2. Download and save the fixed data subsets with reproducible IDs and class counts.
3. Fit a scikit-learn pipeline: TF-IDF word unigrams/bigrams plus logistic regression. Fit the vocabulary only on training data; use validation data for any choices.
4. Fine-tune google-bert/bert-base-uncased with a two-class head, maximum length 128, learning rate 2e-5, and initially one epoch. Use a small batch suited to available memory. Measure a short training benchmark before committing to a full CPU run. Choose settings using validation data only. Record truncation and runtime.
5. Prompt an instruction-following LLM with a fixed zero-shot sentiment instruction. Prefer a locally runnable open model if hardware allows; alternatively use an available authorized API. Record the exact model/version, prompt, decoding settings, raw responses, runtime and invalid outputs. Never use a pretrained sentiment classifier as the prompted LLM. Do not assume API access or invent responses.
6. Evaluate all three on exactly the same 200 held-out reviews: accuracy, macro precision/recall/F1, confusion matrices and timing. Report LLM invalid-response count and coverage, and penalize invalid outputs in overall accuracy rather than silently excluding them.
7. Inspect a few actual disagreements and mistakes, including mixed sentiment and negation. Discuss limited sample size, truncation, compute requirements and possible LLM pretraining contamination. Do not assume BERT or the LLM must beat the baseline.
8. Save the executed notebook, runnable source, dependency versions, prediction records, comparison figures and a concise report. Verify the reported metrics against saved predictions.

## Time estimate
The stated 60 minutes is an assignment budget, not a requirement to spend exactly an hour. A small experiment can fit roughly 45–75 minutes when dependencies, model access and a GPU are ready. Setup and downloads may add 15–45 minutes. CPU-only full BERT fine-tuning may take hours; runtime must be benchmarked. These are planning estimates, not measured timings on this computer.

Indicative 60-minute budget: setup/data 10 minutes; baseline 5; BERT 20; LLM 10; evaluation/report 15. If local training is too slow, prepare everything in this VS Code folder and execute the notebook on an available hosted GPU; external account access may be needed.

## Planned deliverables
- Executed assignment.ipynb with explanations and actual results.
- Reproducible Python implementation and requirements file.
- Per-review predictions and comparison figures.
- report.md covering methodology, measured results, error analysis and limitations.

Keep intermediate data, caches and checkpoints under work/. Put final deliverables under outputs/. No fabricated scores or unexecuted results presented as completed work.

## References
- Dataset: https://huggingface.co/datasets/fancyzhx/amazon_polarity
- Fine-tuning reference: https://huggingface.co/docs/transformers/tasks/sequence_classification
