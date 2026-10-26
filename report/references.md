# References

## Core papers (plan)

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin.
   **Attention Is All You Need.** *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).*
   arXiv:1706.03762.
2. A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. **Transformers are RNNs: Fast Autoregressive
   Transformers with Linear Attention.** *Proceedings of the 37th International Conference on Machine
   Learning (ICML 2020)*, PMLR 119. arXiv:2006.16236.
3. R. Eldan, Y. Li. **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?**
   2023. arXiv:2305.07759.

## Datasets

4. TinyStories dataset — Hugging Face `roneneldan/TinyStories` (accompanies [3]).
   Licence: *to be confirmed by Member 2 from the dataset card and recorded here.*
5. M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, R. Xin.
   **Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM.** Databricks blog, 2023.
   Dataset `databricks/databricks-dolly-15k`. Licence: *to be confirmed by Member 2 from the dataset card
   (published as CC BY-SA 3.0) and recorded here.*

## Methods used in training and evaluation

6. I. Loshchilov, F. Hutter. **Decoupled Weight Decay Regularization.** *ICLR 2019.* arXiv:1711.05101.
   (AdamW optimiser.)
7. J. Li, M. Galley, C. Brockett, J. Gao, B. Dolan. **A Diversity-Promoting Objective Function for Neural
   Conversation Models.** *NAACL-HLT 2016.* arXiv:1510.03055. (distinct-n.)
8. C.-Y. Lin. **ROUGE: A Package for Automatic Evaluation of Summaries.** *Text Summarization Branches
   Out (ACL Workshop), 2004.* (ROUGE-L.)
9. A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi. **The Curious Case of Neural Text Degeneration.**
   *ICLR 2020.* arXiv:1904.09751. (Repetition under greedy decoding; top-k / sampling.)
10. L. Ouyang et al. **Training language models to follow instructions with human feedback.**
    *NeurIPS 2022.* arXiv:2203.02155. (Supervised fine-tuning stage on instruction/response pairs.)

Add any further source here **before** citing it in the report; every entry must be a source a team
member has actually read.
