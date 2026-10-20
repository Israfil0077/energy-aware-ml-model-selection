# Additional Papers for the Literature Review

Compiled 2026-10-04 to strengthen the paper beyond the team's 15-paper table (`literature_review_table.xlsx`).
Every entry below was checked online: title, authors and venue were confirmed on the publisher page, dblp, arXiv or the ACM/IEEE library. Links are given so anyone can re-check.

## Why these were added

The team's 15 papers cover the **motivation** well: data-centre energy growth and AI for data-centre efficiency. Reviewers of this paper will also expect four things the table doesn't yet cover:

1. **The dataset's own citation.** Bitbrains requires it.
2. **Prior work forecasting VM workloads on the same Bitbrains data.** This lets us compare.
3. **"Green AI" work on the energy cost of ML models themselves.** This is our core argument.
4. **Multi-criteria (TOPSIS) model selection, and classical-vs-deep-learning comparisons.** These support our method and our findings.

---

## A. Dataset (required)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| A1 | S. Shen, V. van Beek, A. Iosup, "Statistical Characterization of Business-Critical Workloads Hosted in Cloud Datacenters," *Proc. 15th IEEE/ACM Int. Symp. Cluster, Cloud and Grid Computing (CCGrid)*, 2015, pp. 465–474. | https://dblp.org/rec/conf/ccgrid/ShenBI15.html | **Required citation and acknowledgement** for GWA-T-12 Bitbrains (Section III-A). Note: the authors are Shen, van Beek and Iosup. |

## B. Workload forecasting on Bitbrains / cloud VMs (direct comparison)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| B1 | P. K. Kollu, "Comparative analysis of cloud resources forecasting using deep learning techniques based on VM workload traces," *Trans. Emerging Telecommunications Technologies*, vol. 35, no. 1, e4933, 2024. | https://onlinelibrary.wiley.com/doi/abs/10.1002/ett.4933 | **Closest prior work.** Compares MLP, ARNN, CNN and LSTM for CPU/memory forecasting on the **same GWA-T-12 Bitbrains** trace (1,250 VMs). It finds the simple MLP best, but reports **no energy or training-cost measurements**: exactly our gap. |

## C. Green AI: the energy cost of ML models themselves (core argument)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| C1 | R. Schwartz, J. Dodge, N. A. Smith, O. Etzioni, "Green AI," *Communications of the ACM*, vol. 63, no. 12, pp. 54–63, 2020. doi:10.1145/3381831 | https://cacm.acm.org/research/green-ai/ | Founding argument that efficiency should be an evaluation criterion **alongside accuracy**. Directly motivates our framework. |
| C2 | A. S. Luccioni, Y. Jernite, E. Strubell, "Power Hungry Processing: Watts Driving the Cost of AI Deployment?," *Proc. ACM FAccT*, 2024. doi:10.1145/3630106.3658542 | https://dl.acm.org/doi/10.1145/3630106.3658542 | Systematic energy comparison of ML model types per 1,000 inferences, showing large differences between model families. |
| C3 | T. Vente, L. Wegmeth, A. Said, J. Beel, "From Clicks to Carbon: The Environmental Toll of Recommender Systems," *Proc. 18th ACM RecSys*, 2024. doi:10.1145/3640457.3688074 | https://arxiv.org/abs/2408.08203 | Deep-learning papers emit about 42× more CO₂ than traditional methods, often without better accuracy. Parallels our LSTM finding in another domain. |
| C4 | T. da Silva Barros, F. Giroire, R. Aparicio-Pardo, J. Moulierac, "Small is Sufficient: Reducing the World AI Energy Consumption Through Model Selection," arXiv:2510.01889, 2025. | https://arxiv.org/abs/2510.01889 | Argues that **choosing appropriately sized models** could cut global AI energy by about 28%. Strong support for model selection as a sustainability lever. (Preprint: say so in the reference.) |

## D. Measuring energy and CO₂ (method + Limitations)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| D1 | B. Courty, V. Schmidt, S. Luccioni, et al., "mlco2/codecarbon," Zenodo, 2024. doi:10.5281/zenodo.11171501 | https://doi.org/10.5281/zenodo.11171501 | Software citation for CodeCarbon (Section III, energy tracking). The official citation lists many contributors; IEEE style allows "et al." |
| D2 | L. Bouza, A. Bugeau, L. Lannelongue, "How to estimate carbon footprint when training deep learning models? A guide and review," *Environmental Research Communications*, vol. 5, no. 11, 115014, 2023. doi:10.1088/2515-7620/acf81b | https://iopscience.iop.org/article/10.1088/2515-7620/acf81b | Compares carbon tools; CodeCarbon (machine mode) was the most accurate, at about 80–102% of a wattmeter. Supports our tool choice. |
| D3 | R. Fischer, "Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements," *it – Information Technology*, 2025. doi:10.1515/itit-2025-0031 | https://arxiv.org/abs/2509.22092 | Estimation tools follow the right patterns but can be off by **up to 40%** in absolute terms. Supports our Limitations section: relative comparisons are reliable, absolute kWh less so. |

## E. Data-centre energy context (motivation, peer-reviewed)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| E1 | E. Masanet, A. Shehabi, N. Lei, S. Smith, J. Koomey, "Recalibrating global data center energy-use estimates," *Science*, vol. 367, no. 6481, pp. 984–986, 2020. | https://ui.adsabs.harvard.edu/abs/2020Sci...367..984M/abstract | Peer-reviewed baseline for data-centre energy use. Strengthens the Introduction alongside the EPRI/IFRI reports already in the table. |

## F. Classical vs deep learning (supports the LSTM finding)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| F1 | L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," *NeurIPS Datasets and Benchmarks*, 2022. | https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf | Tree models such as XGBoost stay state of the art on tabular data, even before counting their speed. Explains why LightGBM/XGBoost beat the LSTM on our test set. |
| F2 | A. Zeng, M. Chen, L. Zhang, Q. Xu, "Are Transformers Effective for Time Series Forecasting?," *Proc. AAAI*, vol. 37, pp. 11121–11128, 2023. | https://ojs.aaai.org/index.php/AAAI/article/view/26317 | One-layer linear models beat complex Transformers on 9 benchmarks. Supports our Linear Regression result and our decision not to add a Transformer. |

## G. Multi-criteria (TOPSIS) model selection (supports the method)

| # | Reference | Link | Use in paper |
|---|---|---|---|
| G1 | Z. Koubaa, A. El-Amraoui, A. Frikha, F. Delmotte, "Multicriteria Decision Making for Selecting Forecasting Electricity Demand Models," *Sustainability*, vol. 16, no. 21, 9219, 2024. doi:10.3390/su16219219 | https://www.mdpi.com/2071-1050/16/21/9219 | Uses ANP + **TOPSIS to select forecasting models** from several error measures. The closest methodological precedent. Our work extends it by adding **energy and CO₂** as criteria. |
| G2 | C.-L. Hwang, K. Yoon, *Multiple Attribute Decision Making: Methods and Applications*. Berlin: Springer, 1981. | (standard textbook) | Original TOPSIS method (Section III, TOPSIS procedure). |

## H. Standard method citations (well-known; for Section III)

These are the standard citations for the tools and models we used:
- **Models:** L. Breiman, "Random Forests," *Machine Learning*, 2001; T. Chen, C. Guestrin, "XGBoost," *KDD* 2016; G. Ke et al., "LightGBM," *NeurIPS* 2017; S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
- **Libraries:** F. Pedregosa et al., "Scikit-learn," *JMLR*, 2011; A. Paszke et al., "PyTorch," *NeurIPS* 2019.

They'll be added to `references.bib` at Phase 12, with full details checked then.

---

## Notes on the team's original 15 papers (to fix before citing)

- **Row 7 (Ashiq):** the title given ("Designing and Analyzing Green Networking Solutions in Data Centers Using AI for Energy Optimization") doesn't match the ResearchGate link, which is titled "Green Computing: Advancing Energy-Efficient Data Centers With AI." Please confirm which paper was reviewed.
- **Rows 11–13 (Rafin) and 17–19 (Nur Amin):** years are missing. Needed for the IEEE reference list.
- **Row 18:** arXiv ID 2606.21064. Please confirm the link opens the intended paper; it has a stray "?" at the end.
- **Several entries are reviews, white papers or reports** (EPRI 2024, IFRI 2025, the 2025 AI-powered-solutions review, the 2024 load-balancing paper). They're fine for the Introduction and motivation. For "Related Work" on methods, the peer-reviewed experimental papers above carry more weight with reviewers.
- **Rows 11 and 12** (campus energy, smart-grid blockchain) are about buildings and grids, not data centres. They're best cited briefly as "AI energy forecasting in adjacent domains," or dropped if space is tight.
