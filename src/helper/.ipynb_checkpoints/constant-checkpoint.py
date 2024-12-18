import os
import socket
from pathlib import Path

epsilon = 1e-7

def get_proj_dir():    
    hostname = socket.gethostname()
    dir = Path(__file__).parent.parent.parent
    return dir

def get_spots_csv(seq_):   
    proj_dir = get_proj_dir()
    folder_path = proj_dir / f"data/preprocessed/{seq_}"    
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    path = folder_path / f"spots.csv"      
    return path
  
def get_edges_csv(seq_):   
    proj_dir = get_proj_dir()
    folder_path = proj_dir / f"data/preprocessed/{seq_}"
    
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    path = folder_path / f"edges.csv"       
    return path    


def get_xml_path(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/annotation/{seq_}.xml"
    return path
    
def get_utids_csv(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/preprocessed/{seq_}/utids.csv"
    return path

def get_spots_interp_csv(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/preprocessed/{seq_}/spots_interp.csv"
    return path

def get_spots_vel_csv(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/preprocessed/{seq_}/spots_velocity.csv"
    return path

def get_mitosis_csv(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/preprocessed/{seq_}/mitosis.csv"
    return path

def get_sparse_spots_csv(seq_, sparse_val):
    proj_dir = get_proj_dir()
    sparse_dir = proj_dir / f"data/sparse/{seq_}/sparse_{sparse_val}"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{sparse_dir}' is ready.")
    path = sparse_dir / "spots.csv"
    
    return path

def get_sparse_edges_csv(seq_, sparse_val):
    proj_dir = get_proj_dir()
    sparse_dir = proj_dir / f"data/sparse/{seq_}/sparse_{sparse_val}"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{sparse_dir}' is ready.")
    path = sparse_dir / "edges.csv"
    
    return path

def get_prelap_save_path(seq_,gap):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/preprocessed/{seq_}/prelap_{seq_}_dt{gap}.csv"
    return path

def get_pred_csv_path(seq_,gap):
    proj_dir = get_proj_dir()
    path = proj_dir / f'./data/predictions/pred_{seq_}_sgap{gap}.csv'
    return path

def get_images_path(seq_):
    proj_dir = get_proj_dir()
    path = proj_dir / f"data/datasets/mcf10a/{seq_}"
    return path

def get_marked_images_path(seq_):
    proj_dir = get_proj_dir()
    marked_images_path = proj_dir / f"data/visualization/{seq_}/marked_images"    
    marked_images_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{marked_images_path}' is ready.")
    
    return marked_images_path


def get_videos_path(seq_):
    proj_dir = get_proj_dir()
    videos_path = proj_dir / f"data/visualization/{seq_}/video/"
    videos_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{videos_path}' is ready.")
    
    return videos_path

if __name__ == '__main__':
    print(get_proj_dir())