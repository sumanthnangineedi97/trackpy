import numpy as np
import pandas as pd
import trackpy as tp

# Class for Trackpy-based linking of particle tracks
class TrackpyLinking():
    def __init__(self):
        # Initialize the class with the total number of frames
        self.num_frames = 433

    # Extract and rename relevant columns from the input dataframe
    def get_df_spots_features(self, df_):
        # Select specific columns
        df_ = df_[['ID', 'FRAME', 'track_id_unique', 'track_id_parent', 'pos_x', 'pos_y', 'dist_link', 'ID_next', 'u_flag']]
        
        # Rename columns for Trackpy compatibility
        df_ = df_.rename(columns={'FRAME': 'frame', 'pos_x': 'x', 'pos_y': 'y'})
        return df_

    # Perform normal linking of tracks using Trackpy
    def normal_linking(self, df_, ids_l, ids_r, search_range_):
        # Filter the spots dataframe for relevant IDs
        spots_df = df_
        ids_lr = ids_l + ids_r
        filtered_df = spots_df[spots_df['ID'].isin(ids_lr)]
        
        # Prepare dataframe for Trackpy linking
        filtered_df = self.get_df_spots_features(filtered_df)
        
        try:
            # Perform linking using Trackpy's link_df method
            linked_df = tp.link_df(
                filtered_df, 
                search_range_, 
                adaptive_stop=50, 
                adaptive_step=0.99, 
                memory=0
            )
        except Exception as e:
            print("Error: ", e)
            return [], []

        # Initialize lists for linked row and column indices
        rows_linked = []
        cols_linked = []
        
        # Iterate through left IDs and check linking
        for i in range(len(ids_l)):
            i_particle = linked_df[linked_df['ID'] == ids_l[i]]['particle'].values[0]
            
            # Check if linking was successful
            if len(linked_df[(linked_df['ID'].isin(ids_lr)) & (linked_df['particle'] == i_particle)]['ID'].values) == 1:
                print('No-linking for SpotID: ', ids_l[i])
            else:
                # Append indices of linked rows and columns
                rows_linked.append(i)
                cols_linked.append(
                    ids_r.index(linked_df[(linked_df['ID'].isin(ids_r)) & (linked_df['particle'] == i_particle)]['ID'].values[0])
                )

        return rows_linked, cols_linked

        
        
        
