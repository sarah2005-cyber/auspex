import traceback
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))

def try_import(name):
    print(f"--- Importing {name} ---")
    try:
        mod = __import__(name)
        print(f"OK: imported {name} -> {mod}")
        return True
    except Exception as e:
        print(f"FAILED to import {name}: {e}")
        traceback.print_exc()
        return False


def main():
    # Try importing repo model components
    for name in [
        'AttentionFeatureFusionSPM',
        'LightweightCNNClassifier',
        'EmbeddingRateEstimator',
        'ResidualUnit',
    ]:
        try_import(name)

    # Try importing our backend package pieces
    try_import('backend.model')
    try_import('backend.app')

    # Try instantiating FullPipelineModel without loading checkpoint
    try:
        from backend.model import FullPipelineModel, load_model
        print('\n--- Instantiating FullPipelineModel ---')
        try:
            m = FullPipelineModel()
            print('OK: FullPipelineModel instance created:', type(m))
        except Exception as e:
            print('FAILED to instantiate FullPipelineModel:', e)
            traceback.print_exc()

        print('\n--- Trying load_model with checkpoint paths (repo root and backend/) ---')
        paths = [
            BASE.parent / 'FullPipelineModel_seed42_epoch29_F10.8015_best.pt',
            BASE / 'FullPipelineModel_seed42_epoch29_F10.8015_best.pt'
        ]
        for p in paths:
            print('Trying', p)
            model = load_model(str(p))
            print('-> load_model returned:', model)
    except Exception as e:
        print('Error while testing model instantiation:', e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
