from src.data.visualize_sparse import GenerateVideo
from src.helper.constant import get_spots_csv, get_edges_csv, get_sparse_spots_csv, get_sparse_edges_csv,get_images_path


def generate_video(seq, num_frames, sparse_gap, fps, show_tid, show_next, show_link):
    
    gen_vid = GenerateVideo(seq, num_frames, sparse_gap, fps, show_tid, show_next, show_link)
    gen_vid.pc_mark_spots()  
    gen_vid.pc_make_video()  


if __name__ == '__main__':
    print("Video Generation")

    # Configuration
    seq_id = 'FLD_7'
    num_frames = 433 # Actual number of frames in dataset
    sparse_gap = 8  # Interval for sparse gap in frames
    fps = 1  # Frames per second for the video
    show_tid = True  # Display track ID for each spot
    show_next = False  # Display position of the spot in the next frame
    show_link = True  # Display arrow indicating next position

    # Generate the video
    generate_video(seq_id, num_frames, sparse_gap, fps, show_tid, show_next, show_link)


    