This repository contains the results of research "Interpreting Large Language Models for Genomics Applications".

In order to interpret chosen model architecture (Caduceus) it was fine-tuned on two classification tasks: sequence classification for promoter region prediction and token classification for Z-DNA regions.
We then applied Integrated Gradients, LRP and SHAP methods to the resulting models and compared most important tokens to the scientific literature to ensure biological consistency. We find that Caduceus model assigns token importances correctly, while simple distillation into Mamba model does not produce satisfying results.

You can find source code for promoter and Z-DNA classification tasks in distil-gue.ipynb and Kouzine_hg38-all.ipynb respectively.

