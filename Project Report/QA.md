"How did you handle small clusters?"

Answer: In my custom_resample function, I set k_neighbors to min(n_min, k_neighbors + 1). This prevents the code from crashing if a cluster has fewer than 3 minority samples.

"Is there data leakage in your oversampling?"

Answer: No. My evaluation loop fits the StandardScaler and applies fit_resample only on the training folds within the 10-fold cross-validation. The test fold remains untouched.