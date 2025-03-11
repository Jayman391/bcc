from featurize_text import make_feature_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  label_binarize
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc

def main():
    # Parse command-line argument for the CSV file path.
    parser = argparse.ArgumentParser(
        description="Multiclass classification pipeline for text data with ensemble classifiers."
    )
    parser.add_argument("filepath", type=str, help="Path to the CSV file containing the data.")
    args = parser.parse_args()

    # Load the CSV file.
    df = pd.read_csv(args.filepath)
    
    # Ensure the CSV has a "text" column.
    if "text" not in df.columns:
        raise ValueError("The input CSV must contain a 'text' column.")
    
    # Extract features and targets.
    X = df["text"].tolist()
    y = df["label"].values

    
    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # List of SentenceTransformer model names to evaluate.
    model_names = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-mpnet-base-v2",
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L12-v2",
        "LaBSE",
        "all-roberta-large-v1",
    ]
    
    # Define multiple classifiers and fuse them using soft-voting.
    estimators = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gp', GaussianProcessClassifier()),
        ('ada', AdaBoostClassifier()),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier())
    ]

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble_classifier = OneVsRestClassifier(ensemble)
        
    # Iterate over each SentenceTransformer model.
    for model_name in model_names:
        print(f"\nEvaluating pipeline with SentenceTransformer model: {model_name}")
        
        # Build the complete pipeline with feature extraction and the ensemble classifier.
        pipeline = Pipeline([
            ('features', make_feature_pipeline(modelname=model_name)),
            ('classifier', ensemble_classifier)
        ])
        
        # Fit the pipeline.
        pipeline.fit(X_train, y_train)
        
        # Predict on the test set.
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        
        # Prepare for ROC AUC analysis.
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curves and AUC for each class.
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # get name of file
        import os
        filename = os.path.basename(args.filepath)
        filename = filename.split('_')[0]

        if not os.path.exists(f"output/plots/{filename}"):
            os.mkdir(f"output/plots/{filename}")


        
        # Plot the ROC curves.
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')
       
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"output/plots/{filename}/roc_{model_name}.png")

        # plot confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.savefig(f"output/plots/{filename}/cm_{model_name}.png")

        if not os.path.exists(f"output/models/{filename}"):
            os.mkdir(f"output/models/{filename}")

        # save model
        import joblib
        joblib.dump(pipeline, f"output/models/{filename}/{model_name}.joblib")

if __name__ == "__main__":
    main()
