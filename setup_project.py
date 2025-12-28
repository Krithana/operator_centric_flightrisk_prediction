# setup_project.py
import os

def create_hybrid_structure():
    """Create optimal hybrid project structure"""
    
    structure = {
        'notebooks/': [
            '01_data_exploration.ipynb',
            '02_feature_engineering.ipynb', 
            '03_model_experiments.ipynb',
            '04_results_analysis.ipynb'
        ],
        'src/': [
            '__init__.py',
            'data_collection.py',
            'preprocessing.py', 
            'model_training.py',
            'evaluation.py',
            'utils.py'
        ],
        'data/': [
            'raw/',
            'processed/',
            'external/'
        ],
        'models/': [],
        'papers/': [
            'figures/',
            'main_paper.tex',
            'references.bib'
        ],
        'streamlit/': [
            'app.py',
            'components/__init__.py'
        ],
        'tests/': [],
        'docs/': [],
        '.vscode/': ['settings.json']
    }
    
    for folder, files in structure.items():
        os.makedirs(folder, exist_ok=True)
        for file in files:
            if file.endswith('/'):
                os.makedirs(os.path.join(folder, file), exist_ok=True)
            else:
                # Ensure parent directories exist (handles nested paths like 'components/__init__.py')
                parent_dir = os.path.dirname(os.path.join(folder, file))
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                open(os.path.join(folder, file), 'w').close()
    
    print("✅ Hybrid project structure created!")

if __name__ == "__main__":
    create_hybrid_structure()