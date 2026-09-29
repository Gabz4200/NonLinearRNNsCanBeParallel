"""Upload this repository to Kaggle as a dataset."""

from nonlinearrnnscanbeparallel.scripts.upload_dataset import main, upload_dataset

__all__ = ["upload_dataset", "main"]

if __name__ == "__main__":
    main()
