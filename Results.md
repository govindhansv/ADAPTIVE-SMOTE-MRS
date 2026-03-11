🚀 SMOTE-MRS Base Paper Replication

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]
✅ Pima Diabetes - Random Forest Done
✅ Pima Diabetes - Naïve Bayes Done
✅ Pima Diabetes - SVM Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]
✅ UCI CKD (Kaggle) - Random Forest Done
✅ UCI CKD (Kaggle) - Naïve Bayes Done
✅ UCI CKD (Kaggle) - SVM Done

======================================================================
           BASE PAPER REPLICATION - BOTH DATASETS
======================================================================
         Dataset         Model  Accuracy  Recall     F1    AUC
   Pima Diabetes Random Forest    0.7407  0.6487 0.6350 0.8066
   Pima Diabetes   Naïve Bayes    0.7134  0.6981 0.6288 0.7896
   Pima Diabetes           SVM    0.6939  0.6675 0.6019 0.7735
UCI CKD (Kaggle) Random Forest    0.9900  0.9960 0.9922 1.0000
UCI CKD (Kaggle)   Naïve Bayes    0.9700  0.9520 0.9751 1.0000
UCI CKD (Kaggle)           SVM    0.9925  0.9920 0.9940 1.0000
======================================================================


🚀 ADAPTIVE SMOTE-MRS Implementation

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]
✅ Pima Diabetes - Random Forest Done
✅ Pima Diabetes - Naïve Bayes Done
✅ Pima Diabetes - SVM Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]
✅ UCI CKD (Kaggle) - Random Forest Done
✅ UCI CKD (Kaggle) - Naïve Bayes Done
✅ UCI CKD (Kaggle) - SVM Done

======================================================================
        ADAPTIVE SMOTE-MRS - BOTH DATASETS (IR THRESHOLD = 1.5)
======================================================================
         Dataset         Model  Accuracy  Recall     F1    AUC
   Pima Diabetes Random Forest    0.7499  0.6298 0.6347 0.7993
   Pima Diabetes   Naïve Bayes    0.6964  0.6830 0.6127 0.7695
   Pima Diabetes           SVM    0.7017  0.6789 0.6121 0.7709
UCI CKD (Kaggle) Random Forest    0.9900  0.9960 0.9922 1.0000
UCI CKD (Kaggle)   Naïve Bayes    0.9700  0.9520 0.9751 1.0000
UCI CKD (Kaggle)           SVM    0.9925  0.9920 0.9940 1.0000
======================================================================


🚀 SMOTE-MRS Base Paper - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1

✅ Random Forest Done
✅ Naïve Bayes Done
✅ Logistic Regression Done

============================================================
      BASE PAPER RESULTS - STROKE PREDICTION
============================================================
              Model  Accuracy  Recall     F1    AUC
      Random Forest    0.9252  0.0802 0.0912 0.7988
        Naïve Bayes    0.7115  0.7068 0.1951 0.7657
Logistic Regression    0.7160  0.6347 0.1792 0.7458
============================================================


🚀 ADAPTIVE SMOTE-MRS - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1

✅ Random Forest Done
✅ Naïve Bayes Done
✅ Logistic Regression Done

============================================================
   ADAPTIVE SMOTE-MRS RESULTS - STROKE PREDICTION
           (IR THRESHOLD = 1.5)
============================================================
              Model  Accuracy  Recall     F1    AUC
      Random Forest    0.9233  0.0842 0.0960 0.7983
        Naïve Bayes    0.7125  0.7188 0.1989 0.7663
Logistic Regression    0.7182  0.6307 0.1793 0.7460
============================================================


🚀 XGBoost with Base SMOTE-MRS - Pima & CKD

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]

📊 Pima Diabetes - Scale Pos Weight: 1.87
  Fold 1: Recall=0.8519, Accuracy=0.7922
  Fold 2: Recall=0.6667, Accuracy=0.7532
  Fold 3: Recall=0.6667, Accuracy=0.6623
  Fold 4: Recall=0.7407, Accuracy=0.8182
  Fold 5: Recall=0.7778, Accuracy=0.7403
  Fold 6: Recall=0.7037, Accuracy=0.7143
  Fold 7: Recall=0.6667, Accuracy=0.6623
  Fold 8: Recall=0.7778, Accuracy=0.7662
  Fold 9: Recall=0.7692, Accuracy=0.7500
  Fold 10: Recall=0.5385, Accuracy=0.6711
✅ Pima Diabetes - XGBoost Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]

📊 UCI CKD (Kaggle) - Scale Pos Weight: 1.67
  Fold 1: Recall=1.0000, Accuracy=0.9750
  Fold 2: Recall=1.0000, Accuracy=1.0000
  Fold 3: Recall=1.0000, Accuracy=1.0000
  Fold 4: Recall=1.0000, Accuracy=1.0000
  Fold 5: Recall=1.0000, Accuracy=1.0000
  Fold 6: Recall=0.9200, Accuracy=0.9500
  Fold 7: Recall=1.0000, Accuracy=0.9500
  Fold 8: Recall=1.0000, Accuracy=1.0000
  Fold 9: Recall=1.0000, Accuracy=1.0000
  Fold 10: Recall=1.0000, Accuracy=1.0000
✅ UCI CKD (Kaggle) - XGBoost Done

======================================================================
        XGBoost with BASE SMOTE-MRS - BOTH DATASETS
======================================================================
         Dataset   Model  Accuracy  Recall     F1    AUC
   Pima Diabetes XGBoost    0.7330   0.716 0.6516 0.7887
UCI CKD (Kaggle) XGBoost    0.9875   0.992 0.9900 0.9989
======================================================================


🚀 XGBoost with ADAPTIVE SMOTE-MRS - Pima & CKD

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]

📊 Pima Diabetes - Scale Pos Weight: 1.87
  Fold 1: Recall=0.8148, Accuracy=0.7532
  Fold 2: Recall=0.6667, Accuracy=0.7403
  Fold 3: Recall=0.5556, Accuracy=0.6623
  Fold 4: Recall=0.7037, Accuracy=0.8182
  Fold 5: Recall=0.7407, Accuracy=0.7532
  Fold 6: Recall=0.6296, Accuracy=0.7013
  Fold 7: Recall=0.6667, Accuracy=0.7403
  Fold 8: Recall=0.7778, Accuracy=0.7792
  Fold 9: Recall=0.6923, Accuracy=0.7105
  Fold 10: Recall=0.5385, Accuracy=0.6447
✅ Pima Diabetes - XGBoost (Adaptive) Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]

📊 UCI CKD (Kaggle) - Scale Pos Weight: 1.67
  Fold 1: Recall=1.0000, Accuracy=0.9750
  Fold 2: Recall=1.0000, Accuracy=1.0000
  Fold 3: Recall=1.0000, Accuracy=1.0000
  Fold 4: Recall=1.0000, Accuracy=1.0000
  Fold 5: Recall=1.0000, Accuracy=1.0000
  Fold 6: Recall=0.9200, Accuracy=0.9500
  Fold 7: Recall=1.0000, Accuracy=0.9500
  Fold 8: Recall=1.0000, Accuracy=1.0000
  Fold 9: Recall=1.0000, Accuracy=1.0000
  Fold 10: Recall=1.0000, Accuracy=1.0000
✅ UCI CKD (Kaggle) - XGBoost (Adaptive) Done

======================================================================
     XGBoost with ADAPTIVE SMOTE-MRS - BOTH DATASETS
         (IR THRESHOLD = 1.5)
======================================================================
         Dataset   Model  Accuracy  Recall     F1    AUC
   Pima Diabetes XGBoost    0.7303  0.6786 0.6367 0.7877
UCI CKD (Kaggle) XGBoost    0.9875  0.9920 0.9900 0.9997
======================================================================


🚀 XGBoost with Base SMOTE-MRS - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1


📊 Stroke Prediction Dataset
  Fold 1: Recall=0.0400, Accuracy=0.9061
  Fold 2: Recall=0.0800, Accuracy=0.9100
  Fold 3: Recall=0.1200, Accuracy=0.9119
  Fold 4: Recall=0.0800, Accuracy=0.9159
  Fold 5: Recall=0.0800, Accuracy=0.9217
  Fold 6: Recall=0.1600, Accuracy=0.9237
  Fold 7: Recall=0.0800, Accuracy=0.9393
  Fold 8: Recall=0.2800, Accuracy=0.9295
  Fold 9: Recall=0.2000, Accuracy=0.9256
  Fold 10: Recall=0.1250, Accuracy=0.9041
✅ Stroke Prediction - XGBoost (Base) Done

============================================================
   XGBoost with BASE SMOTE-MRS - STROKE PREDICTION
============================================================
  Model  Accuracy  Recall     F1    AUC
XGBoost    0.9188  0.1245 0.1296 0.7839
============================================================



🚀 XGBoost with ADAPTIVE SMOTE-MRS - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1


📊 Stroke Prediction Dataset
  Fold 1: Recall=0.0400, Accuracy=0.9178
  Fold 2: Recall=0.0800, Accuracy=0.9080
  Fold 3: Recall=0.1600, Accuracy=0.9159
  Fold 4: Recall=0.1200, Accuracy=0.9178
  Fold 5: Recall=0.1200, Accuracy=0.9159
  Fold 6: Recall=0.1200, Accuracy=0.9022
  Fold 7: Recall=0.1200, Accuracy=0.9315
  Fold 8: Recall=0.2000, Accuracy=0.9256
  Fold 9: Recall=0.2400, Accuracy=0.9276
  Fold 10: Recall=0.2083, Accuracy=0.9061
✅ Stroke Prediction - XGBoost (Adaptive) Done

============================================================
 XGBoost with ADAPTIVE SMOTE-MRS - STROKE PREDICTION
     (IR THRESHOLD = 1.5)
============================================================
  Model  Accuracy  Recall     F1    AUC
XGBoost    0.9168  0.1408 0.1407 0.7937
===========================================================



🚀 LightGBM with Base SMOTE-MRS - Pima & CKD

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]

📊 Pima Diabetes - Scale Pos Weight: 1.87
  Fold 1: Recall=0.6667, Accuracy=0.7143
  Fold 2: Recall=0.5926, Accuracy=0.7532
  Fold 3: Recall=0.5185, Accuracy=0.6753
  Fold 4: Recall=0.7037, Accuracy=0.8052
  Fold 5: Recall=0.6296, Accuracy=0.7273
  Fold 6: Recall=0.5926, Accuracy=0.7273
  Fold 7: Recall=0.7037, Accuracy=0.7273
  Fold 8: Recall=0.6667, Accuracy=0.7532
  Fold 9: Recall=0.6154, Accuracy=0.7500
  Fold 10: Recall=0.4615, Accuracy=0.6579
✅ Pima Diabetes - LightGBM Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]

📊 UCI CKD (Kaggle) - Scale Pos Weight: 1.67
  Fold 1: Recall=1.0000, Accuracy=0.9750
  Fold 2: Recall=1.0000, Accuracy=1.0000
  Fold 3: Recall=1.0000, Accuracy=1.0000
  Fold 4: Recall=1.0000, Accuracy=1.0000
  Fold 5: Recall=1.0000, Accuracy=1.0000
  Fold 6: Recall=0.9600, Accuracy=0.9750
  Fold 7: Recall=0.9600, Accuracy=0.9500
  Fold 8: Recall=1.0000, Accuracy=1.0000
  Fold 9: Recall=0.9600, Accuracy=0.9750
  Fold 10: Recall=1.0000, Accuracy=1.0000
✅ UCI CKD (Kaggle) - LightGBM Done

======================================================================
       LightGBM with BASE SMOTE-MRS - BOTH DATASETS
======================================================================
         Dataset    Model  Accuracy  Recall     F1    AUC
   Pima Diabetes LightGBM    0.7291  0.6151 0.6121 0.7840
UCI CKD (Kaggle) LightGBM    0.9875  0.9880 0.9900 0.9992
======================================================================


🚀 LightGBM with ADAPTIVE SMOTE-MRS - Pima & CKD

📊 Pima Diabetes: 768 samples, Class distribution: [500 268]

📊 Pima Diabetes
  Fold 1: Recall=0.7407, Accuracy=0.7532
  Fold 2: Recall=0.5926, Accuracy=0.7532
  Fold 3: Recall=0.4815, Accuracy=0.6753
  Fold 4: Recall=0.6667, Accuracy=0.8312
  Fold 5: Recall=0.6667, Accuracy=0.7532
  Fold 6: Recall=0.6296, Accuracy=0.7273
  Fold 7: Recall=0.7037, Accuracy=0.7013
  Fold 8: Recall=0.7407, Accuracy=0.7792
  Fold 9: Recall=0.6154, Accuracy=0.7500
  Fold 10: Recall=0.5000, Accuracy=0.6842
✅ Pima Diabetes - LightGBM (Adaptive) Done

📊 UCI CKD (Kaggle): 400 samples, Class distribution: [150 250]

📊 UCI CKD (Kaggle)
  Fold 1: Recall=1.0000, Accuracy=0.9750
  Fold 2: Recall=1.0000, Accuracy=1.0000
  Fold 3: Recall=1.0000, Accuracy=1.0000
  Fold 4: Recall=1.0000, Accuracy=1.0000
  Fold 5: Recall=1.0000, Accuracy=1.0000
  Fold 6: Recall=0.9600, Accuracy=0.9750
  Fold 7: Recall=0.9600, Accuracy=0.9750
  Fold 8: Recall=1.0000, Accuracy=1.0000
  Fold 9: Recall=0.9600, Accuracy=0.9750
  Fold 10: Recall=1.0000, Accuracy=1.0000
✅ UCI CKD (Kaggle) - LightGBM (Adaptive) Done

======================================================================
    LightGBM with ADAPTIVE SMOTE-MRS - BOTH DATASETS
         (IR THRESHOLD = 1.5)
======================================================================
         Dataset    Model  Accuracy  Recall     F1    AUC
   Pima Diabetes LightGBM    0.7408  0.6338 0.6295 0.7894
UCI CKD (Kaggle) LightGBM    0.9900  0.9880 0.9919 0.9992
======================================================================



🚀 LightGBM with Base SMOTE-MRS - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1


📊 Stroke Prediction Dataset
  Fold 1: Recall=0.1200, Accuracy=0.9335
  Fold 2: Recall=0.0400, Accuracy=0.9198
  Fold 3: Recall=0.1600, Accuracy=0.9295
  Fold 4: Recall=0.0400, Accuracy=0.9256
  Fold 5: Recall=0.0800, Accuracy=0.9178
  Fold 6: Recall=0.1200, Accuracy=0.9374
  Fold 7: Recall=0.0800, Accuracy=0.9374
  Fold 8: Recall=0.1200, Accuracy=0.9315
  Fold 9: Recall=0.1600, Accuracy=0.9256
  Fold 10: Recall=0.1250, Accuracy=0.9276
✅ Stroke Prediction - LightGBM (Base) Done

============================================================
   LightGBM with BASE SMOTE-MRS - STROKE PREDICTION
============================================================
   Model  Accuracy  Recall     F1    AUC
LightGBM    0.9286  0.1045 0.1244 0.7874
============================================================


🚀 LightGBM with ADAPTIVE SMOTE-MRS - Stroke Prediction Dataset

📊 Samples: 5110
📊 Class distribution: [4861  249]
📊 Imbalance Ratio: 19.52:1


📊 Stroke Prediction Dataset
  Fold 1: Recall=0.0800, Accuracy=0.9198
  Fold 2: Recall=0.0400, Accuracy=0.9256
  Fold 3: Recall=0.1600, Accuracy=0.9354
  Fold 4: Recall=0.0400, Accuracy=0.9276
  Fold 5: Recall=0.0400, Accuracy=0.9256
  Fold 6: Recall=0.1200, Accuracy=0.9354
  Fold 7: Recall=0.0800, Accuracy=0.9413
  Fold 8: Recall=0.1200, Accuracy=0.9393
  Fold 9: Recall=0.2000, Accuracy=0.9276
  Fold 10: Recall=0.1250, Accuracy=0.9295
✅ Stroke Prediction - LightGBM (Adaptive) Done

============================================================
 LightGBM with ADAPTIVE SMOTE-MRS - STROKE PREDICTION
     (IR THRESHOLD = 1.5)
============================================================
   Model  Accuracy  Recall     F1    AUC
LightGBM    0.9307  0.1005 0.1225 0.7916
============================================================