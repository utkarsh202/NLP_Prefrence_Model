from setuptools import setup, find_packages

setup(
    name="nlp_preference_model",
    version="0.1.0",
    description="Reliable Pairwise Preference Modeling for LLM Responses (IIT Bhilai)",
    author="Utkarsh",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4",
        "transformers>=4.44",
        "datasets>=2.20",
        "scikit-learn>=1.4",
        "scipy",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "streamlit",
    ],
)
