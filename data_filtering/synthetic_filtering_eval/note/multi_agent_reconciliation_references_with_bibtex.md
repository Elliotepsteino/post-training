# Reconciling Conflicting Outputs from Multiple LLM Agents — Research References

You can reconcile three (or more) disagreeing agent answers using **(a) aggregation**, **(b) judge-based selection**, **(c) interactive deliberation (debate / critique / peer review)**, and **(d) verification/consistency checks**. Below are research papers and surveys directly relevant to these strategies, with short summaries and links.

---

## 1) Simple aggregation (vote / self-consistency)

### Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022)
- **Reference:** Xuezhi Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. 2022.  
- **Link:** https://arxiv.org/abs/2203.11171  
- **Summary:** Instead of taking one greedy chain-of-thought, the model samples **multiple reasoning paths** and then selects the most consistent final answer via **marginalization/majority vote** over sampled answers. This is one of the most widely used and simplest ways to reconcile divergent outputs when the task has a well-defined final answer (e.g., math/logic QA).

- **BibTeX:**
```bibtex
@misc{wang2022selfconsistency,
  title        = {Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author       = {Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc V. and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  year         = {2022},
  eprint       = {2203.11171},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2203.11171},
  url          = {https://arxiv.org/abs/2203.11171}
}
```


### Ranked Voting based Self-Consistency of Large Language Models (2025)
- **Reference:** Weiqin Wang, Yile Wang, and Hui Huang. *Ranked Voting based Self-Consistency of Large Language Models*. Findings of ACL 2025.  
- **Link:** https://aclanthology.org/2025.findings-acl.744.pdf  
- **Summary:** Extends classic self-consistency by using **ranked voting** ideas so the aggregation step can better account for alternative answers that would be missed by single-answer-per-sample voting. Useful when candidates are diverse and “near-miss” answers exist.

- **BibTeX:**
```bibtex
@inproceedings{wang-etal-2025-ranked,
  title     = {Ranked Voting based Self-Consistency of Large Language Models},
  author    = {Wang, Weiqin and Wang, Yile and Huang, Hui},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  month     = jul,
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.findings-acl.744/},
  doi       = {10.18653/v1/2025.findings-acl.744},
  pages     = {14410--14426},
  isbn      = {979-8-89176-256-5}
}
```


---

## 2) Judge-based selection (LLM-as-a-Judge / pairwise or listwise ranking)

### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (2023)
- **Reference:** Lianmin Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. 2023.  
- **Link:** https://arxiv.org/abs/2306.05685  
- **Summary:** Provides evidence that strong LLMs can act as **judges** whose preferences often align with humans (in their benchmarks). In practice, this supports the pattern: *generate multiple candidate answers → ask a judge model to select the best*.

- **BibTeX:**
```bibtex
@misc{zheng2023judging,
  title        = {Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author       = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  year         = {2023},
  eprint       = {2306.05685},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2306.05685},
  url          = {https://arxiv.org/abs/2306.05685}
}
```


### G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (2023)
- **Reference:** Yang Liu et al. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023.  
- **Link:** https://arxiv.org/abs/2303.16634  
- **Summary:** Proposes a structured way to use an LLM for evaluation (chain-of-thought + form-filling). While designed for NLG evaluation, the same approach is commonly adapted to **score and select** among multiple agent outputs with explicit criteria (helpfulness, correctness, faithfulness, etc.).

- **BibTeX:**
```bibtex
@misc{liu2023geval,
  title        = {{G}-{E}val: {NLG} Evaluation using {GPT}-4 with Better Human Alignment},
  author       = {Liu, Yang and Liu, Lixin and Hu, Chenxu and Yuan, Wei and Wu, Aixing and Mou, Lili and Yu, Philip S.},
  year         = {2023},
  eprint       = {2303.16634},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2303.16634},
  url          = {https://arxiv.org/abs/2303.16634}
}
```


### A Survey on LLM-as-a-Judge (2024)
- **Reference:** Jiawei Gu et al. *A Survey on LLM-as-a-Judge*. 2024.  
- **Link:** https://arxiv.org/abs/2411.15594  
- **Summary:** Surveys how to use LLMs as evaluators in **pointwise, pairwise, and listwise** settings, including typical prompting patterns, bias/failure modes, and evaluation considerations. Helpful for designing robust “arbiter/judge” pipelines.

- **BibTeX:**
```bibtex
@misc{gu2024llmasajudge_survey,
  title        = {A Survey on LLM-as-a-Judge},
  author       = {Gu, Jiawei and Jiang, Xuhui and Shi, Zhichao and Tan, Hexiang and Zhai, Xuehao and Xu, Chengjin and Li, Wei and Shen, Yinghan and Ma, Shengjie and Liu, Honghao and Wang, Saizhuo and Zhang, Kun and Wang, Yuanzhuo and Gao, Wen and Ni, Lionel and Guo, Jian},
  year         = {2024},
  eprint       = {2411.15594},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2411.15594},
  url          = {https://arxiv.org/abs/2411.15594}
}
```


### JUDGEBENCH: A Benchmark for Evaluating LLM-Based Judges (ICLR 2025)
- **Reference:** Sijun Tan et al. *JUDGEBENCH: A Benchmark for Evaluating LLM-Based Judges*. ICLR 2025.  
- **Link:** https://openreview.net/pdf?id=G0dksFayVq  
- **Summary:** Focuses on the **reliability of judge models** themselves. This is important because reconciliation pipelines that rely on a judge can inherit judge biases or inconsistencies; this paper provides a benchmark and framing for measuring those weaknesses.

- **BibTeX:**
```bibtex
@inproceedings{tan2025judgebench,
  title        = {JudgeBench: A Benchmark for Evaluating LLM-based Judges},
  author       = {Tan, Sijun and Zhuang, Siyuan and Montgomery, Kyle and Tang, William Y. and Cuadron, Alejandro and Wang, Chenguang and Popa, Raluca Ada and Stoica, Ion},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2025},
  eprint       = {2410.12784},
  archivePrefix= {arXiv},
  doi          = {10.48550/arXiv.2410.12784},
  url          = {https://openreview.net/forum?id=G0dksFayVq}
}
```


---

## 3) Interactive deliberation (agents talk to reconcile)

### Improving Factuality and Reasoning in Language Models through Multiagent Debate (2023)
- **Reference:** Yilun Du et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. 2023.  
- **Link:** https://arxiv.org/abs/2305.14325  
- **Summary:** Multiple model instances propose answers and **debate over multiple rounds** to converge to a final answer. The paper reports improved reasoning and factuality versus single-pass generation on several tasks, motivating the “debate → consensus” reconciliation pattern.

- **BibTeX:**
```bibtex
@misc{du2023multiagentdebate,
  title        = {Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author       = {Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B. and Mordatch, Igor},
  year         = {2023},
  eprint       = {2305.14325},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2305.14325},
  url          = {https://arxiv.org/abs/2305.14325}
}
```


### Debating with More Persuasive LLMs Leads to More Truthful Answers (2024)
- **Reference:** Akbir Khan et al. *Debating with More Persuasive LLMs Leads to More Truthful Answers*. 2024.  
- **Link:** https://arxiv.org/abs/2402.06782  
- **Summary:** Studies debate settings where a judge (human or model) decides after seeing arguments. Explores how debate can help identify truth and how optimizing for persuasion affects outcomes—useful background when designing debate-based reconciliation.

- **BibTeX:**
```bibtex
@misc{khan2024debating,
  title        = {Debating with More Persuasive LLMs Leads to More Truthful Answers},
  author       = {Khan, Akbir and Hughes, John and Valentine, Dan and Ruis, Laura and Sachan, Kshitij and Radhakrishnan, Ansh and Grefenstette, Edward and Bowman, Samuel R. and Rockt{\"a}schel, Tim and Perez, Ethan},
  year         = {2024},
  eprint       = {2402.06782},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2402.06782},
  url          = {https://arxiv.org/abs/2402.06782}
}
```


### Mixture-of-Agents Enhances Large Language Model Capabilities (ICLR 2025)
- **Reference:** Junlin Wang et al. *Mixture-of-Agents Enhances Large Language Model Capabilities*. ICLR 2025.  
- **Link:** https://proceedings.iclr.cc/paper_files/paper/2025/file/5434be94e82c54327bb9dcaf7fca52b6-Paper-Conference.pdf  
- **Summary:** Proposes a **layered architecture**: first-layer agents independently generate responses; later-layer agents **see all previous answers** and produce a refined output. This is a direct “agents exchange answers → refine” reconciliation mechanism.

- **BibTeX:**
```bibtex
@inproceedings{wang2025mixtureofagents,
  title        = {Mixture-of-Agents Enhances Large Language Model Capabilities},
  author       = {Wang, Junlin and Wang, Jue and Athiwaratkun, Ben and Zhang, Ce and Zou, James},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2025},
  eprint       = {2406.04692},
  archivePrefix= {arXiv},
  doi          = {10.48550/arXiv.2406.04692},
  url          = {https://arxiv.org/abs/2406.04692}
}
```


### Multi-Agent Debate for LLM Judges with Adaptive Stability (2025)
- **Reference:** Tianyu Hu et al. *Multi-Agent Debate for LLM Judges with Adaptive Stability Detection*. 2025.  
- **Link:** https://arxiv.org/abs/2510.12697  
- **Summary:** Targets the case where you already use LLMs as judges but simple aggregation (e.g., majority vote) can fail. Proposes having judges **debate and iteratively refine** their judgments, with analysis intended to improve stability over static ensembles.

- **BibTeX:**
```bibtex
@misc{hu2025multiagentdebate_judges,
  title        = {Multi-Agent Debate for LLM Judges with Adaptive Stability Detection},
  author       = {Hu, Tianyu and Tan, Zhen and Wang, Song and Qu, Huaizhi and Chen, Tianlong},
  year         = {2025},
  eprint       = {2510.12697},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2510.12697},
  url          = {https://arxiv.org/abs/2510.12697}
}
```


---

## 4) Critique-and-refine loops (interaction via critique, not necessarily debate)

### Self-Refine: Iterative Refinement with Self-Feedback (2023)
- **Reference:** Aman Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023.  
- **Link:** https://arxiv.org/abs/2303.17651  
- **Summary:** Shows a simple loop: **generate → critique → refine → repeat** using (often) the same model. While not “multi-agent” by default, it’s a foundation for multi-agent reconciliation where each agent critiques others and then a final agent synthesizes revisions.

- **BibTeX:**
```bibtex
@misc{madaan2023selfrefine,
  title        = {Self-Refine: Iterative Refinement with Self-Feedback},
  author       = {Madaan, Aman and Tandon, Niket and Gupta, Prakhar and Hallinan, Skyler and Gao, Luyu and Wiegreffe, Sarah and Alon, Uri and Dziri, Nouha and Prabhumoye, Shrimai and Yang, Yiming and Gupta, Shashank and Majumder, Bodhisattwa Prasad and Hermann, Katherine and Welleck, Sean and Yazdanbakhsh, Amir and Clark, Peter},
  year         = {2023},
  eprint       = {2303.17651},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2303.17651},
  url          = {https://arxiv.org/abs/2303.17651}
}
```


### Training Language Model to Critique for Better Refinement (RCO) (2025)
- **Reference:** Tianyu Yu et al. *Training Language Model to Critique for Better Refinement*. Findings of ACL 2025.  
- **Link:** https://aclanthology.org/2025.findings-acl.1373.pdf  
- **Summary:** Studies what makes critiques useful and proposes a training framework to optimize critique quality for downstream refinement. If your reconciliation method depends on critique quality, this is directly relevant.

- **BibTeX:**
```bibtex
@inproceedings{yu-etal-2025-training,
  title     = {Training Language Model to Critique for Better Refinement},
  author    = {Yu, Tianshu and Xiang, Chao and Yang, Mingchuan and Ke, Pei and Wen, Bosi and Wang, Cunxiang and Cheng, Jiale and Zhang, Li and Mu, Xinyu and Sun, Chuxiong and Huang, Minlie},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  month     = jul,
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.findings-acl.1373/},
  doi       = {10.18653/v1/2025.findings-acl.1373},
  pages     = {26760--26804},
  isbn      = {979-8-89176-256-5}
}
```


### DeepCritic: Deliberate Critique with Large Language Models (2025)
- **Reference:** Wenkai Yang et al. *DeepCritic: Deliberate Critique with Large Language Models*. 2025.  
- **Link:** https://arxiv.org/abs/2505.00662  
- **Summary:** Introduces a two-stage approach to produce **step-by-step, deeper critiques** (initial critiques can be shallow). Particularly relevant when reconciling multi-step reasoning outputs (math proofs, code, plans).

- **BibTeX:**
```bibtex
@misc{yang2025deepcritic,
  title        = {DeepCritic: Deliberate Critique with Large Language Models},
  author       = {Yang, Wenkai and Chen, Jingwen and Lin, Yankai and Wen, Ji-Rong},
  year         = {2025},
  eprint       = {2505.00662},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2505.00662},
  url          = {https://arxiv.org/abs/2505.00662}
}
```


### CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing (ICLR 2024)
- **Reference:** Zhibin Gou et al. *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*. ICLR 2024.  
- **Link:** https://proceedings.iclr.cc/paper_files/paper/2024/file/fef126561bbf9d4467dbb8d27334b8fe-Paper-Conference.pdf  
- **Summary:** Proposes having the model use **external tools** (e.g., search/calculators) to critique and revise outputs. In reconciliation pipelines, this supports the pattern: *multiple candidates → critique with tools → choose/synthesize grounded answer*.

- **BibTeX:**
```bibtex
@inproceedings{gou2024critic,
  title        = {CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing},
  author       = {Gou, Zhibin and Shao, Zhihong and Gong, Yeyun and Shen, Yelong and Yang, Yujiu and Duan, Nan and Chen, Weizhu},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2024},
  eprint       = {2305.11738},
  archivePrefix= {arXiv},
  doi          = {10.48550/arXiv.2305.11738},
  url          = {https://openreview.net/forum?id=Sx038qxjek}
}
```


### Can Large Language Models Really Improve by Self-critiquing Their Own Plans? (2023)
- **Reference:** Karthik Valmeekam et al. *Can Large Language Models Really Improve by Self-critiquing Their Own Plans?* 2023.  
- **Link:** https://arxiv.org/abs/2310.08118  
- **Summary:** A cautionary result: self-critiquing can **degrade** performance in planning when verification is unreliable (false positives). This motivates using (i) stronger external verifiers, (ii) multi-judge aggregation, or (iii) debate/critique with grounding, rather than trusting a single self-critic.

- **BibTeX:**
```bibtex
@misc{valmeekam2023selfcritiqueplans,
  title        = {Can Large Language Models Really Improve by Self-critiquing Their Own Plans?},
  author       = {Valmeekam, Karthik and Marquez, Matthew and Kambhampati, Subbarao},
  year         = {2023},
  eprint       = {2310.08118},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2310.08118},
  url          = {https://arxiv.org/abs/2310.08118}
}
```


---

## 5) Search / structured exploration over multiple candidates (generate-many → evaluate)

### Tree of Thoughts: Deliberate Problem Solving with Large Language Models (NeurIPS 2023)
- **Reference:** Shunyu Yao et al. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS 2023.  
- **Link:** https://arxiv.org/abs/2305.10601  
- **Summary:** Generalizes chain-of-thought to a **tree search over “thoughts”**: generate multiple partial solutions, evaluate them, and expand the most promising branches. This is a systematic reconciliation mechanism when you can score intermediate states (not just final answers).

- **BibTeX:**
```bibtex
@misc{yao2023treeofthoughts,
  title        = {Tree of Thoughts: Deliberate Problem Solving with Large Language Models},
  author       = {Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak and Griffiths, Thomas L. and Cao, Yuan and Narasimhan, Karthik},
  year         = {2023},
  eprint       = {2305.10601},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2305.10601},
  url          = {https://arxiv.org/abs/2305.10601}
}
```


---

## 6) LLM ensemble / routing (choose among agent outputs, sometimes adaptively)

### Harnessing Multiple Large Language Models: A Survey on LLM Ensemble (2025)
- **Reference:** Zhijun Chen et al. *Harnessing Multiple Large Language Models: A Survey on LLM Ensemble*. 2025.  
- **Link:** https://arxiv.org/abs/2502.18036  
- **Summary:** A broad taxonomy of LLM ensemble strategies: **before inference** (routing), **during inference** (token/step-level aggregation), and **after inference** (selection/aggregation). Great map of the space and pointers to many additional papers.

- **BibTeX:**
```bibtex
@misc{chen2025llmensemble_survey,
  title        = {Harnessing Multiple Large Language Models: A Survey on LLM Ensemble},
  author       = {Chen, Zhijun and Li, Jingzheng and Chen, Pengpeng and Li, Zhuoran and Sun, Kai and Luo, Yuankai and Mao, Qianren and Li, Ming and Xiao, Likang and Yang, Dingqi and Ban, Yikun and Sun, Hailong and Yu, Philip S.},
  year         = {2025},
  eprint       = {2502.18036},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2502.18036},
  url          = {https://arxiv.org/abs/2502.18036}
}
```


### Smoothie: Label Free Language Model Routing (NeurIPS 2024)
- **Reference:** Neel Guha et al. *Smoothie: Label Free Language Model Routing*. NeurIPS 2024.  
- **Link:** https://arxiv.org/abs/2412.04692  
- **Summary:** An unsupervised method that, given a set of candidate outputs, learns to **route/select** which model’s output to use without labeled data (using weak-supervision style latent variable modeling over embeddings). Relevant for reconciling multiple agents by *selecting the best source* rather than merging.

- **BibTeX:**
```bibtex
@misc{guha2024smoothie,
  title        = {Smoothie: Label Free Language Model Routing},
  author       = {Guha, Neel and Chen, Mayee F. and Chow, Trevor and Khare, Ishan S. and R{\'e}, Christopher},
  year         = {2024},
  eprint       = {2412.04692},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2412.04692},
  url          = {https://arxiv.org/abs/2412.04692}
}
```


### Efficient Dynamic Ensembling for Multiple LLM Experts (IJCAI 2025)
- **Reference:** Jinwu Hu et al. *Efficient Dynamic Ensembling for Multiple LLM Experts*. IJCAI 2025.  
- **Link:** https://www.ijcai.org/proceedings/2025/0900.pdf  
- **Summary:** Frames ensembling as a **sequential decision process**: an agent decides which expert LLM to query next and passes information forward. Relevant when you want interaction like: *ask agent A → ask agent B with A’s answer → ask agent C with both → finalize*.

- **BibTeX:**
```bibtex
@inproceedings{ijcai2025p900,
  title     = {Efficient Dynamic Ensembling for Multiple LLM Experts},
  author    = {Hu, Jinwu and Wang, Yufeng and Zhang, Shuhai and Zhou, Kai and Chen, Guohao and Hu, Yu and Xiao, Bin and Tan, Mingkui},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {8095--8103},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/900},
  url       = {https://doi.org/10.24963/ijcai.2025/900}
}
```


### Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process (2025/2026)
- **Reference:** Zhijun Chen et al. *Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process* (LLM-PeerReview). 2025/2026.  
- **Link:** https://arxiv.org/abs/2512.23213  
- **Summary:** Proposes an unsupervised “peer review” pipeline: multiple models generate candidates; multiple models score them as judges; then scores are aggregated (including reliability-aware aggregation) and the best response is selected. This directly matches the “multiple agents disagree → peer review → pick winner” reconciliation story.

- **BibTeX:**
```bibtex
@article{chen2025scoring,
  title   = {Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process},
  author  = {Chen, Zhijun and Ji, Zeyu and Mao, Qianren and Wu, Hao and Cheng, Junhang and Qin, Bangjie and Li, Zhuoran and Li, Jingzheng and Sun, Kai and Wang, Zizhe and Ban, Yikun and Sun, Zhu and Ji, Xiangyang and Sun, Hailong},
  journal = {arXiv preprint arXiv:2512.23213},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.23213},
  url     = {https://arxiv.org/abs/2512.23213}
}
```


---

## 7) Consistency-based verification (detect hallucinations / unreliable claims)

### SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection (2023)
- **Reference:** Potsawee Manakul et al. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*. EMNLP 2023.  
- **Link:** https://arxiv.org/abs/2303.08896  
- **Summary:** Uses multiple stochastic samples from the same (or similar) model and measures **inconsistency/divergence** to flag likely hallucinations. In multi-agent reconciliation, this supports a pattern like: *if agents disagree strongly on factual claims → trigger retrieval/tool verification or abstain.*

- **BibTeX:**
```bibtex
@inproceedings{manakul-etal-2023-selfcheckgpt,
  title     = {{S}elf{C}heck{GPT}: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author    = {Manakul, Potsawee and Liusie, Adian and Gales, Mark},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year      = {2023},
  month     = dec,
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.emnlp-main.557/},
  doi       = {10.18653/v1/2023.emnlp-main.557},
  pages     = {9004--9017}
}
```


---

## Practical reconciliation templates for 3 agents (inspired by the papers above)

1. **Self-consistency / voting (fastest)**
   - Run each agent (possibly with different temperatures/prompts) → majority vote on the final answer (or on extracted structured fields).  
   - Best when the answer is discrete and objectively checkable.

2. **Judge-based selection**
   - Ask the three agents for answers → feed the candidates to a **judge prompt** (pairwise tournament or listwise ranking) → select winner.  
   - Grounded by MT-Bench / LLM-as-a-judge literature.

3. **Debate → judge**
   - Round 1: each agent proposes answer + reasons.  
   - Round 2: each agent critiques the others (or a structured “attack/defend” format).  
   - Final: a judge or synthesizer produces the final answer.  
   - Grounded by multi-agent debate work.

4. **Mixture-of-Agents (layered refinement)**
   - Layer 1: 3 parallel answers.  
   - Layer 2: a “refiner” agent sees all 3 and produces a consolidated answer (optionally with citations/verification).

5. **Peer-review scoring**
   - Agents generate candidates → each agent also scores all candidates against a rubric → aggregate scores → pick final.  
   - Inspired by LLM-PeerReview.

---

### Notes / caveats
- Reconciliation works best when you define **what “better” means** (correctness, completeness, safety, style, citation requirements).
- Judge/critic reliability is not guaranteed; see JudgeBench and planning self-critique work for failure modes.
