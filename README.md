# Domain Adaptation as Simple as a Linear Function

**Arpit Sahni, Aviral Chharia, Burhanuddin Shirose, Srinivas Gowriraj**

**TA Led Project:** Hira Dhamyal, PhD Candidate and Prof. Bhiksha Raj

**Abstract**

Deep Neural Networks (DNNs) that are trained on a particular source domain may not necessarily perform well on other target domains that share similar properties due to differences in their feature space. Therefore, domain adaptation has gained attention in recent years as a means of leveraging knowledge from a source domain to improve model performance on a target domain. In this study, we propose a generalized domain adaptation technique that can be applied to both speech and visual modalities. Our approach involves introducing a direct affine transformation that brings the target domain into the same latent space as the source domain, which allows models trained on large datasets to adapt to new domains. Our universal domain-adaptation layer works across modalities and learning methods. We evaluate the performance of our technique on benchmark datasets in both speech and vision and compare it with standard baselines. Our experiments have demonstrated that the proposed method outperforms conventional fully connected and linear layers for domain adaptation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235063593-8c982f74-0023-4a26-8552-b5f232c519b3.png" />
</p>

**Ablations**

![image](https://user-images.githubusercontent.com/62457915/235062877-938db197-4e4c-4012-878b-12b2d0fce244.png)

**Implementation Workbench**

The work is implemented using PyTorch. NVIDIA Geforce RTX-3090 Ti with 24GB RAM workbench was used for conducting the experiments.
