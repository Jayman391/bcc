from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA

import umap
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# Transformer to generate embeddings using SentenceTransformer.
class SentenceTransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, modelname="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(modelname)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, list):
            X = X.tolist()
        embeddings = self.model.encode(X)
        return np.array(embeddings)

# Transformer that wraps HashingVectorizer to return a dense array.
class DenseHashingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=128, alternate_sign=False):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                            alternate_sign=self.alternate_sign)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

# Transformer for VADER sentiment analysis.
class VaderSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            scores = self.analyzer.polarity_scores(text)
            features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        return np.array(features)

# UMAP transformer.
class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=16, **umap_kwargs):
        self.n_components = n_components
        self.umap_kwargs = umap_kwargs
        self.reducer = umap.UMAP(n_components=self.n_components, **self.umap_kwargs)
    
    def fit(self, X, y=None):
        self.reducer.fit(X)
        return self
    
    def transform(self, X):
        return self.reducer.transform(X)
    



def make_feature_pipeline(modelname="all-MiniLM-L6-v2", umap_components=4):
    # Pipeline for SentenceTransformer embeddings with UMAP reduction.
    st_pipeline = Pipeline([
        ("sentence_transformer", SentenceTransformerFeaturizer(modelname=modelname)),
        ("pca", PCA(n_components=64)),
        ("umap_st", UMAPTransformer(n_components=umap_components))
    ])
    
    # Pipeline for Hashing vectorizer features with UMAP reduction.
    hash_pipeline = Pipeline([
        ("hash_vectorizer", DenseHashingVectorizer(n_features=1024)),
        ("pca", PCA(n_components=64)),
        ("umap_hash", UMAPTransformer(n_components=umap_components))
    ])
    
    # Pipeline for VADER sentiment (no UMAP reduction required).
    vader_pipeline = Pipeline([
        ("vader_sentiment", VaderSentimentTransformer())
    ])
    
    # Combine the pipelines.
    combined_features = FeatureUnion(transformer_list=[
        ("st_pipeline", st_pipeline),
        ("hash_pipeline", hash_pipeline),
        ("vader_pipeline", vader_pipeline)
    ])
    
    # Build the complete pipeline.
    pipeline = Pipeline([
        ("features", combined_features)
    ])
    
    return pipeline
