# Title Slide

## Adaptive SMOTE-MRS: An Enhanced Multi-Resolution Sampling Technique for Imbalanced Medical Data Classification

#### Good morning, respected teachers and examiners. My name is Govind Hans V, and today I will be presenting my final year project titled: 'Adaptive SMOTE-MRS: An Enhanced Multi-Resolution Sampling Technique for Imbalanced Medical Data Classification'. This work was conducted under the guidance of Ms. Anjana T.K. at the Department of Computer Science, St. Joseph's College, Devagiri

Sir/Ma'am, my title essentially means I am using a clustering-based approach (Multi-Resolution) to find specific pockets of Imbalanced Data. I am Enhancing the existing framework by making it Adaptive, meaning the system intelligently decides whether or not to generate Synthetic Data (SMOTE) based on the specific needs of each local cluster, rather than applying a blanket solution to the entire dataset.


# Abstract 

The main challenge I addressed is that standard models often ignore minority disease cases because of Class Imbalance. While methods like SMOTE exist, they are often 'blind' and introduce Noise by oversampling areas that don't need it. My project proposes an Adaptive version that checks the Imbalance Ratio of each cluster first. This ensured that we only intervened where necessary, leading to a 13.1% boost in identifying high-risk stroke patients.


In medical diagnosis, we rely on machine learning to detect life-threatening conditions early. However, real-world medical data is inherently imbalanced. This means we have a massive number of 'Healthy' cases but very few 'Disease' cases. Standard models are designed to maximize overall accuracy, so they often 'ignore' the rare disease cases just to get a high score. In a hospital, this is dangerous because missing one sick patient is a much bigger failure than a false alarm


Traditional ways to fix this involve oversampling—specifically a technique called SMOTE. SMOTE creates synthetic or 'fake' examples of the minority class so the model has more to learn from. The base paper for my work, titled 'SMOTE-MRS', improved this by first grouping the data into clusters before applying SMOTE, which helps keep the synthetic data more organized

While the base paper was innovative, it has a significant limitation: it is static. It applies the same amount of oversampling to every cluster, regardless of whether that group actually needs it. This 'blanket' approach often introduces noise—basically garbage data—into regions that are already well-balanced. It lacks a mechanism to focus its efforts only on the truly difficult, imbalanced groups


My project, Adaptive SMOTE-MRS, enhances this by adding an intelligent 'brain' to the process. Instead of oversampling everything, my system calculates the Imbalance Ratio of each cluster first and only intervenes where a genuine imbalance exists



# Literature review

How to Categorize these for FacultyIf the faculty asks you to summarize the literature, group them like this:Foundational Methods (1, 2, 3): These are the "global" oversamplers like standard SMOTE and ADASYN. They work on the whole dataset and often create noise.Clustering Methods (4, 6, 10): These group data first, like K-Means SMOTE and your base paper. Their weakness is that they treat all clusters the same, even the balanced ones.Complex/Deep Methods (7, 8, 9): These use Neural Networks or GANs. They are powerful but are "black-boxes" (hard for doctors to understand) and require too much data.


Now, I will briefly cover the Literature Review. I analyzed 10 major works in this field to identify the current limitations in handling imbalanced data.

"Global vs. Local: "Foundational works like the original SMOTE (2002) and ADASYN (2008) were revolutionary but they operate globally across the whole dataset. This often leads to 'bridging' where the AI creates noisy samples near outliers."


The Clustering Evolution: "To fix this, researchers introduced clustering-based methods like Cluster-SMOTE and the base paper for my project, SMOTE-MRS (2024). These methods group similar patients together before oversampling."

The Primary Gap: "However, even these advanced methods have a shared drawback: they apply uniform oversampling. They don't check if a cluster needs balancing. This lack of an 'Adaptive' mechanism is exactly what my project addresses by calculating a cluster-level Imbalance Ratio.


Decision Boundary


Simple Explanation: The imaginary line the AI draws to separate "Sick" from "Healthy".

Analogy: Like a fence between two yards. If you put the fence in the wrong place because of "noisy" data, your AI will misclassify who owns which land.

2. Outliers


Simple Explanation: Data points that are very far away from the rest of the group.

Analogy: A healthy person who accidentally has one high blood pressure reading. If the AI oversamples this "outlier," it might start thinking everyone with one high reading is sick.

3. Black-Box Model


Simple Explanation: A system where you can see the input and the output, but you don't know why it made that choice.

Why it matters: In medicine, doctors need to know why a patient is flagged as high-risk. My model uses interpretable logic (Random Forest/XGBoost) instead of a black-box.