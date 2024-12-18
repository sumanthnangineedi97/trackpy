# Import necessary modules

from src.data.preprocess import Preprocessor
from src.helper.constant import get_spots_csv, get_edges_csv, get_sparse_spots_csv, get_sparse_edges_csv

def gen_sparse():
    # Configuration
    sequence_id = 'FLD_7'
    sparse_interval = 8

    # Get file paths for the CSVs
    spots_csv_path = get_spots_csv(sequence_id)
    edges_csv_path = get_edges_csv(sequence_id)
    sparse_spots_csv_path = get_sparse_spots_csv(sequence_id, sparse_interval)
    sparse_edges_csv_path = get_sparse_edges_csv(sequence_id, sparse_interval)

    # Initialize Preprocessor and run preprocessing functions
    preprocessor = Preprocessor()
    preprocessor.gen_sparse_edges(edges_csv_path, sparse_edges_csv_path, sparse_interval)
    preprocessor.gen_sparse_spots(spots_csv_path, sparse_spots_csv_path, sparse_edges_csv_path, sparse_interval)

if __name__ == '__main__':
    gen_sparse()
