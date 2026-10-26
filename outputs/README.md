# Run the assignment in Colab

1. Open https://colab.research.google.com/ and upload Amazon_Review_Classification_Colab.ipynb.
2. Select Runtime > Change runtime type > TPU.
3. Run only the installation cell in section 1. Its TARGET defaults to TPU.
4. When it finishes, select Runtime > Restart session. Continue from section 2, running cells in order. Do not rerun installation.
5. Wait for BERT training and Qwen prompting to finish. The first TPU batch includes compilation and may take several minutes.
6. The final sections generate the metrics, plots, error examples, report and a downloadable results ZIP.
7. Save the notebook with its outputs using File > Download > Download .ipynb. Add your name/student ID before submission.

The notebook is standalone: it downloads its own public dataset sample and pretrained models. No API key or paid LLM service is required. TPU availability depends on the Colab allocation. A single TPU device is used for BERT; Qwen runs on the runtime's CPU. For a GPU runtime, set TARGET='GPU' in section 1 and repeat setup/restart. If the pinned PyTorch/XLA version does not support the allocated TPU, use the GPU option.

## What is complete

- Runnable notebook code for all three methods, comparison, and report export.
- Local data download and TF-IDF baseline executed successfully: 165/200 correct (82.5% accuracy), macro-F1 0.82489.
- Three actual BERT training batches executed on the laptop: approximately 9 seconds per batch, projecting about 38 minutes for the full training epoch before evaluation. This is an estimate, not a completed BERT experiment.
- Notebook syntax/schema and local data/baseline/evaluation/report checks performed.

## What still requires Colab execution

Full BERT training, LLM inference and final three-model scores have not been executed. TPU runtime execution is unverified locally. The notebook intentionally contains no invented outputs. Run it to finish the experimental part and produce the submission report.

## Files

- Amazon_Review_Classification_Colab.ipynb: primary deliverable; upload this alone.
- experiment.py: optional local CPU/CUDA stage runner (not the TPU notebook).
- requirements.txt: optional local Python dependencies.
- data.json: exact locally sampled reviews, source row IDs and split labels.
- tfidf.json: measured local baseline predictions and metrics.
- benchmark.json: measured local three-batch BERT timing and projection.
- PLAN.md: original planning document; this README records the final Colab handoff.

The local script supports data, baseline, benchmark, bert and llm stages; for example `python experiment.py baseline` from the outputs folder. Its files and the Colab notebook are separate execution routes. The notebook includes reporting in addition to those model stages.

## Fix for XLA undefined CUDA symbol

The corrected installation cell explicitly installs torch 2.8.0+cpu from the PyTorch CPU index with torch_xla 2.8.0. Run that cell, restart the Colab session, then run section 2 onward. A CPU PyTorch wheel still supports TPU through XLA. This addresses the reported XLA 2.8/CUDA binary-symbol issue; remote TPU execution remains unverified here. See https://github.com/pytorch/xla/issues/9656.
