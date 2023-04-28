# Domain Adaptation as Simple as a Linear Function

**Arpit Sahni, Aviral Chharia, Burhanuddin Shirose, Srinivas Gowriraj**

**TA Led Project:** Hira Dhamyal, PhD Candidate and Prof. Bhiksha Raj

## **Abstract**

Deep Neural Networks (DNNs) that are trained on a particular source domain may not necessarily perform well on other target domains that share similar properties due to differences in their feature space. Therefore, domain adaptation has gained attention in recent years as a means of leveraging knowledge from a source domain to improve model performance on a target domain. In this study, we propose a generalized domain adaptation technique that can be applied to both speech and visual modalities.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235063593-8c982f74-0023-4a26-8552-b5f232c519b3.png" />
</p>

Our approach involves introducing a direct affine transformation that brings the target domain into the same latent space as the source domain, which allows models trained on large datasets to adapt to new domains. Our universal domain-adaptation layer works across modalities and learning methods. We evaluate the performance of our technique on benchmark datasets in both speech and vision and compare it with standard baselines. Our experiments have demonstrated that the proposed method outperforms conventional fully connected and linear layers for domain adaptation.

## **Overall Pipeline**

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235063897-9bf5490c-e6b4-4af7-90ef-75cd102c5658.png" />
</p>

## **Ablations**

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235062877-938db197-4e4c-4012-878b-12b2d0fce244.png" />
</p>

## **Key  Results**

**1. Proposed Sparse linear adaptation achieves superior performance (+13.3% wrt Zero-Shot & +91.4% wrt Linear). Conventional linear adaption shows large performance drop (-40.2% wrt Zero-Shot).**

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235064395-ed15a57f-33c2-4bbc-a8e3-fc4c6002090c.png" />
</p>

**2. Sparse linear adaptation results in decision boundary that is robust to imbalance classes (+32.65% Avg.) in high dimensional feature space.**

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235064476-f49580fe-bec7-4824-9010-e1f5517888b2.png" />
</p>

**3. Linear adaptation degrades the performance on the Source Domain. However, Sparse Linear maintains the same performance.**

<p align="center">
  <img src="https://user-images.githubusercontent.com/62457915/235064531-a8e96596-9f6d-4f1a-9ccb-f173225c5ecf.png" />
</p>

## **Implementation Workbench**

The work is implemented using PyTorch. NVIDIA Geforce RTX-3090 Ti with 24GB RAM workbench was used for conducting the experiments.
