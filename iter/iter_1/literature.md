# T2I Adversarial Attack Papers - Literature Review

## Citation Counts (as of Dec 2024)

| # | Paper | Venue | Year | Citations | Notes |
|---|-------|-------|------|-----------|-------|
| 1 | **SneakyPrompt** | IEEE S&P | 2024 | 201 | First automated RL-based jailbreak framework |
| 2 | **Ring-A-Bell** | ICLR | 2024 | 169 | Model-agnostic red-teaming tool |
| 3 | **MMA-Diffusion** | CVPR | 2024 | 145 | Multimodal attack (text + image) |
| 4 | **Visual-RolePlay** | arXiv | 2024 | 57 | Role-playing based jailbreak for MLLMs |
| 5 | **JPA (Jailbreaking Prompt Attack)** | NAACL | 2025 | 56 | Universal, no target model needed |
| 6 | **SurrogatePrompt** | ACM CCS | 2024 | 39 | 88% bypass on Midjourney |
| 7 | **Perception-guided Jailbreak (PGJ)** | AAAI | 2025 | 35 | Black-box, model-free |
| 8 | **Chain-of-Jailbreak** | arXiv | 2024 | 11 | Step-by-step editing attack |
| 9 | **AdvI2I** | arXiv | 2024 | 5 | Image-based attack (not prompt) |

**Total citations across papers: 718**

---

## Key Observations

1. **Most cited**: SneakyPrompt (201) and Ring-A-Bell (169) dominate - both from top-tier venues (IEEE S&P, ICLR)
2. **Rising impact**: MMA-Diffusion (CVPR 2024) at 145 citations - first multimodal attack
3. **Venue matters**: Top security/ML venues (S&P, ICLR, CVPR, CCS, AAAI, NAACL) get more citations
4. **Research velocity**: Even 2024 arXiv preprints have 11-57 citations

---

## Full Paper Titles

| Attack | Full Title | Authors |
|--------|------------|---------|
| **SneakyPrompt** | *Sneakyprompt: Jailbreaking text-to-image generative models* | Y Yang, B Hui, H Yuan, N Gong, Y Cao |
| **Ring-A-Bell** | *Ring-A-Bell! How Reliable are Concept Removal Methods for Diffusion Models?* | YL Tsai, CY Hsu, C Xie, et al. |
| **MMA-Diffusion** | *Mma-diffusion: Multimodal attack on diffusion models* | Y Yang, R Gao, X Wang, TY Ho, N Xu, Q Xu |
| **Visual-RolePlay** | *Visual-roleplay: Universal jailbreak attack on multimodal large language models via role-playing image character* | S Ma, W Luo, Y Wang, X Liu |
| **JPA** | *Jailbreaking prompt attack: A controllable adversarial attack against diffusion models* | J Ma, Y Li, Z Xiao, A Cao, J Zhang, C Ye, J Zhao |
| **SurrogatePrompt** | *Surrogateprompt: Bypassing the safety filter of text-to-image models via substitution* | Z Ba, J Zhong, J Lei, P Cheng, Q Wang, Z Qin, Z Wang, K Ren |
| **PGJ** | *Perception-guided jailbreak against text-to-image models* | Y Huang, L Liang, T Li, X Jia, R Wang, W Miao, G Pu, Y Liu |
| **Chain-of-Jailbreak** | *Chain-of-jailbreak attack for image generation models via editing step by step* | W Wang, K Gao, Y Yuan, J Huang, Q Liu, et al. |
| **AdvI2I** | *Advi2i: Adversarial image attack on image-to-image diffusion models* | Y Zeng, Y Cao, B Cao, Y Chang, J Chen, L Lin |

---

## Attack Taxonomy

### Prompt-based Attacks
- **SneakyPrompt**: RL-guided token perturbation to bypass safety filters
- **SurrogatePrompt**: LLM-based substitution of sensitive prompt sections
- **Ring-A-Bell**: Steering vectors representing unsafe concepts as optimization targets
- **JPA**: Exploits NSFW concepts in high-dimensional text embedding space
- **PGJ**: Black-box, identifies safe phrases with inconsistent semantics

### Image-based Attacks
- **AdvI2I**: Adversarial image manipulation (not prompt) to bypass I2I model defenses
- **MMA-Diffusion**: Combined text + image perturbations

### Editing-based Attacks
- **Chain-of-Jailbreak**: Decomposes malicious query into harmless sub-queries + iterative editing

### Role-play Attacks
- **Visual-RolePlay**: Embeds harmful content via role-playing image characters in MLLMs

---

## Defense Methods (for reference)

| Defense | Venue | Approach |
|---------|-------|----------|
| **GuardT2I** | NeurIPS 2024 | LLM "translates" adversarial prompts to reveal intent |
| **Espresso** | - | Fine-tuned CLIP classifier for content filtering |
| **Safe Latent Diffusion (SLD)** | - | Latent-space filtering |

---

## Sources

- [SneakyPrompt - IEEE](https://ieeexplore.ieee.org)
- [Ring-A-Bell - Semantic Scholar](https://www.semanticscholar.org/paper/04983bbf48ab9649e3e6dcb7f4fadd7d04c89bbd)
- [MMA-Diffusion - CVPR](https://openaccess.thecvf.com)
- [SurrogatePrompt - ACM CCS](https://dl.acm.org/doi/10.1145/3658644.3690346)
- [JPA - ACL Anthology](https://aclanthology.org)
- [PGJ - AAAI](https://ojs.aaai.org)
- [AdvI2I - arXiv](https://arxiv.org/abs/2410.21471)
- [Awesome Multimodal Jailbreak](https://github.com/liuxuannan/Awesome-Multimodal-Jailbreak)
- [Awesome AD on T2IDM](https://github.com/datar001/Awesome-AD-on-T2IDM)
