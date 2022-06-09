# Main references
* Attention is all you need (2017)
    - https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
* Transformer Tutorials:
    - https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
    - https://jalammar.github.io/illustrated-transformer/
    - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - https://peterbloem.nl/blog/transformers

# Masterplan
1. Train transformer model on translation tasks for fixed number of epochs
2. Apply pruning algorithms
3. Analyze BLEU score before and after pruning

# To Do
* [x] Check BLEU Evaluation for
    - [x] Pretrained, but not finetuned T5-small -> sacrebleu: 18,5 (on WMT16's validation set)
    - [x] Pretrained, but not finetuned T5-base -> sacrebleu: 19.8 (on WMT16's validation set)
    - [x] Pretrained, but not finetuned T5-large -> Does not terminate
    - [x] Pretrained and finetuned for 30 Epochs -> see logs sent to @zimmer-m
* [x] Initialize T5 without pretrained parameters
    - [x] How to train from scratch? -> see config 
    - [x] Train randomly initialized T5 from scratch for 5 Epochs
        - [x] Check BLEU score -> very slow progress
* [ ] Prepare Training for 4x NVIDIA A100 80GB GPUs
    - [ ] Find paper which pretrains on WMT16
    - [ ] Implementation in PyTorch (drop lightning for now)
    - [ ] Define appropriate training, evaluation and test dataset
    - [ ] Define appropriate training routine, i.e., warmup, learning_rate, lr_scheduler, batch_size, epochs etc.