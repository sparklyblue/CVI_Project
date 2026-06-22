"""Compatibility entrypoint for the pairwise crop CNN package.

The implementation lives in pairwise_crop_cnn/ so the code remains split into
smaller modules, similar to the motion baseline package.
"""

from pairwise_crop_cnn.cli import main


if __name__ == "__main__":
    main()

# python train_pairwise_crop_cnn.py --device cuda --epochs 8 --neighbors 1 --crop-size 96 --batch-size 512 --num-workers 8 --tune-metric macro_f1 --overfit-penalty 0.15 --threshold-step 0.02 --output-dir dist/pairwise_crop_cnn
