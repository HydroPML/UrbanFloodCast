# UrbanFloodCast

## Urban flood forecasting benchmark dataset
![UrbanFloodCastBench](https://github.com/HydroPML/UrbanFloodCast/blob/main/Figures/Figures02.png)
Urban flood forecasting benchmark dataset is introduced for evaluating various neural urban flood modeling and simulation methods. Two urban areas in Berlin, Germany are considered. For both areas, simulations are conducted using 125 distinct design storm rainfall events. The dataset is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.15700880).
### Dataset Structure

UrbanFloodCast_Dataset
- Berlin I
  - Input_BerlinI
  - Seen regions and unseen rainfall events
    - Train
    - Valid
    - Test
  - Seen regions and unseen rainfall distributions
    - Train
    - Valid
    - Test
  - Zero-shot downscaling
    - Valid
    - Test
- Berlin II
  - Input_BerlinII
  - EulerII
  - EulerIII
  - Rand
## Deep Neural Operator (DNO)
![DNO](https://github.com/HydroPML/UrbanFloodCast/blob/main/Figures/Figures11.png)
A deep neural operator (DNO) is designed for fast, accurate, and resolution-invariant urban flood forecasting. The DNO features an enhanced Fourier layer with skip connections for improved memory efficiency, alongside a deep encoder-decoder framework and an urban-embedded residual loss to enhance modeling effectiveness. 
### Usage
To use the DNO, execute the following steps:
   
1. **Download traning, validation, and test data. Put in the corresponding folder.**
   
2. **Please use DNO/TIFF2PT to convert TIFF files into PT files, thereby streamlining model training and data loading processes.**

3. **Please change the corresponding path in DNO/DNO_main.py to your path.**

4. **Run the DNO/DNO_main.py to train and test your model.**
## Deep Neural Operator (DNO)
![TL-DNO](https://github.com/HydroPML/UrbanFloodCast/blob/main/Figures/Figures011.png)
To enhance transferability for cross-scenario urban flood modeling and forecasting, we develop a transfer learning-based DNO (TL-DNO). Specifically, we propose two strategies: a fine-tuning-based DNO for efficient prediction in the target domain, and a domain adaptation-based DNO for continuous learning across domains. The fine-tuning-based DNO model is first trained on the source domain and then fine-tuned by adjusting a subset of layers while keeping the remaining layers fixed. This method enables **rapid and accurate adaptation** to cross-scenario target domains and improves the interpretability of the DNO layers. However, the fine-tuning-based DNO can lead to the forgetting of source domain knowledge. To overcome this challenge, we introduce the domain adaptation-based DNO that leverages the adversarial learning framework of generative adversarial networks (GANs) to learn domain-invariant representations from limited labeled target data. This marks the first application of domain adaptation (DA) in NOs for urban flood forecasting, ensuring accurate predictions across diverse domains. The proposed domain adaptation-based DNO comprises a generator for generating multi-scale features and a discriminator to distinguish the domain of each input feature. The DNO serves as the generator. Adversarial learning is employed to align the final multi-scale feature map produced by the generator. To enhance optimization stability and effectiveness, we introduce a progressive training strategy for adversarial learning. Compared with fine-tuning-based approach, domain adaptation-based DNO exhibits greater robustness across both source and target domains, thereby enabling **continuous learning** for diverse urban flood modeling and forecasting tasks.
### Usage
To use the TL-DNO model, execute the following steps:
1. **Download traning, validation, and test data. Put in the corresponding folder.**
   
2. **Please use TL-DNO/TIFF2PT to convert TIFF files into PT files, thereby streamlining model training and data loading processes.**

3. **Please change the corresponding path to your path.**

4. **Run the TL-DNO/Fine-tuning-based_DNO.py to rapid and accurate cross-scenario urban flood forecasting.**

5. **Run the TL-DNO/Domain-adaptation-based_DNO.py to continuous learning for diverse urban flood modeling and forecasting tasks.**  
Please note that options 4 and 5 depend on your specific requirements.






