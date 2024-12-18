from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET 
import math
from tqdm import tqdm

np.random.seed(1988)

from src.helper.constant import get_utids_csv, get_spots_csv,get_prelap_save_path, get_spots_vel_csv

class Preprocessor():
    def spots_xml_to_csv(self, xml_path_, spot_csv_):
        """
        Converts spots and tracks data from an XML file to a CSV file.

        Args:
            xml_path_ (str): Path to the input XML file.
            spot_csv_ (str): Path to the output CSV file.
        Returns:
            int: 0 on successful completion.
        """
        
        
        # Parse the XML file
        print("Parsing XML file...")
        data_tree = ET.parse(xml_path_)
        data_root = data_tree.getroot()

        # Extract spots and tracks
        spots_in_frame = data_root[1][1]
        tracks = data_root[1][2]

        # Define columns for the DataFrame
        columns = [
            'TRACK_ID', 'FRAME', 'ID', 'name', 'QUALITY', 'POSITION_T', 
            'MAX_INTENSITY', 'MEDIAN_INTENSITY', 'VISIBILITY', 
            'MEAN_INTENSITY', 'TOTAL_INTENSITY', 'ESTIMATED_DIAMETER', 
            'RADIUS', 'SNR', 'POSITION_X', 'POSITION_Y', 
            'STANDARD_DEVIATION', 'CONTRAST', 'MANUAL_COLOR', 
            'MIN_INTENSITY', 'POSITION_Z'
        ]
        spots_df = pd.DataFrame(columns=columns)

        # Process each spot in each frame
        print("Processing frames...")
        for spots in tqdm(spots_in_frame, desc="Processing Frames", unit="frame", leave=False, dynamic_ncols=True):
            for spot in spots:
                track_id = -1

                # Find the associated track ID by checking SPOT_SOURCE_ID and SPOT_TARGET_ID
                for track in tracks:
                    for edge in track:
                        if edge.attrib['SPOT_SOURCE_ID'] == spot.attrib['ID']:
                            track_id = track.attrib['TRACK_ID']
                            break

                if track_id == -1:
                    for track in tracks:
                        for edge in track:
                            if edge.attrib['SPOT_TARGET_ID'] == spot.attrib['ID']:
                                track_id = track.attrib['TRACK_ID']
                                break

                # Add TRACK_ID to the spot attributes
                spot.attrib['TRACK_ID'] = track_id

                # Append the spot data to the DataFrame if TRACK_ID exists
                if track_id != -1:
                    df_dict = pd.DataFrame([spot.attrib])
                    spots_df = pd.concat([spots_df, df_dict])


        # Save the DataFrame to a CSV file
        print(f"Saving processed data to CSV: {spot_csv_}")
        spots_df.to_csv(spot_csv_, index=False)
        
        print("Processing completed successfully!")
        return 0
    
    def edges_xml_to_csv(self, xml_path_, edge_csv_):
        """
        Converts edge data from an XML file to a CSV file.

        Args:
            xml_path_ (str): Path to the input XML file.
            edge_csv_ (str): Path to the output CSV file.
        Returns:
            int: 0 on successful completion.
        """
        
        
        # Parse the XML file
        print("Parsing XML file...")
        data_tree = ET.parse(xml_path_)
        data_root = data_tree.getroot()

        # Extract spots and tracks
        spots_in_frame = data_root[1][1]
        tracks = data_root[1][2]

        # Define columns for the DataFrame
        columns = ['FRAME', 'TRACK_ID', 'SPOT_SOURCE_ID', 'SPOT_TARGET_ID']
        edges_df = pd.DataFrame(columns=columns)

        # Process each edge in each track
        print("Processing tracks and edges...")
        #for track in tracks:
        for track in tqdm(tracks, desc="Processing Tracks", unit="track", leave=False, dynamic_ncols=True):
            for edge in track:
                # Merge track attributes with edge attributes
                track.attrib.update(edge.attrib)
                frame = 1111  # Default frame if not found

                # Find the frame by matching the SPOT_SOURCE_ID with spot IDs
                for spots in spots_in_frame:
                    for spot in spots:
                        if spot.attrib['ID'] == edge.attrib['SPOT_SOURCE_ID']:
                            frame = spot.attrib['FRAME']
                            break

                # Add the frame to the attributes
                track.attrib['FRAME'] = frame

                # Filter the attributes to match the desired columns
                filtered_attrib = {k: track.attrib.get(k) for k in columns}

                # Append the data to the DataFrame
                df_dict = pd.DataFrame([filtered_attrib])
                edges_df = pd.concat([edges_df, df_dict], ignore_index=True)

        # Save the DataFrame to a CSV file
        print(f"Saving processed edges data to CSV: {edge_csv_}")
        edges_df.to_csv(edge_csv_, index=False)
        
        print("Processing completed successfully!")
        return 0

    def gen_sparse_edges(self, edge_csv_, sparse_edge_csv_, sparse_val):
        """
        Generate sparse edges by sampling at intervals based on the sparse value.

        Args:
            edge_csv_ (str): Path to the input edges CSV file.
            sparse_edge_csv_ (str): Path to save the output sparse edges CSV file.
            sparse_val (int): The interval value for generating sparse edges.

        Returns:
            int: 0 on successful completion.
        """
        
        
        print(f"Generating sparse edges with sparse value {sparse_val}...")
        
        # Define the columns for the output DataFrame
        columns = ['FRAME', 'TRACK_ID', 'SPOT_SOURCE_ID', 'SPOT_TARGET_ID']
        edges_df = pd.read_csv(edge_csv_)

        # Sort the DataFrame
        print("Sorting edges data...")
        edges_df.sort_values(
            by=['TRACK_ID', 'FRAME', 'SPOT_SOURCE_ID', 'SPOT_TARGET_ID'], 
            inplace=True, 
            ignore_index=True
        )

        # Get the unique track IDs
        unique_track_ids = list(set(edges_df['TRACK_ID']))
        new_edges_df = pd.DataFrame(columns=columns)

        # Process each track ID
        for track_id in unique_track_ids:
            # Filter data for the current track ID
            track_df = edges_df.loc[edges_df['TRACK_ID'] == track_id].reset_index(drop=True)
            start_frame = track_df['FRAME'].iloc[0]
            end_frame = track_df['FRAME'].iloc[-1]

            # Determine the adjusted start frame
            adjusted_start_frame = (start_frame // sparse_val) * sparse_val
            #print('Adjusted Start Frame:', adjusted_start_frame)

            if start_frame % sparse_val > 0:
                if (adjusted_start_frame + sparse_val) < end_frame:
                    adjusted_start_frame += sparse_val
                else:
                    adjusted_start_frame = end_frame

            # Skip processing if the adjusted start frame is the same as the end frame
            if adjusted_start_frame != end_frame:
                for frame_id in range(adjusted_start_frame, end_frame, sparse_val):
                    #print('Processing Frame ID:', frame_id)

                    # Get the unique source spot IDs for the current frame
                    source_ids = track_df.loc[track_df['FRAME'] == frame_id, 'SPOT_SOURCE_ID'].unique()

                    # Skip the first frame
                    if frame_id == adjusted_start_frame:
                        continue

                    # Process each source ID
                    for source_id in source_ids:
                        edge_data = {
                            'SPOT_TARGET_ID': source_id,
                            'TRACK_ID': track_id,
                            'FRAME': (frame_id - sparse_val) // sparse_val
                        }
                        current_source_id = source_id
                        backtrack_count = sparse_val

                        # Backtrack to find the appropriate source ID
                        while backtrack_count > 0:
                            if len(track_df.loc[track_df['SPOT_TARGET_ID'] == current_source_id]) > 0:
                                current_source_id = track_df.loc[
                                    track_df['SPOT_TARGET_ID'] == current_source_id, 
                                    'SPOT_SOURCE_ID'
                                ].values[0]
                                backtrack_count -= 1
                            else:
                                break

                        # Update the source ID if backtracking is successful
                        if backtrack_count == 0:
                            edge_data['SPOT_SOURCE_ID'] = current_source_id

                        # Add the new edge data to the DataFrame if valid
                        if current_source_id != source_id:
                            new_edges_df = pd.concat([new_edges_df, pd.DataFrame([edge_data])], ignore_index=True)

        # Save the sparse edges DataFrame to a CSV file
        print(f"Saving sparse edges to: {sparse_edge_csv_}")
        new_edges_df.to_csv(sparse_edge_csv_, index=False)
        
        print("Sparse edges generation completed successfully.")
        return 0

    def gen_sparse_spots(self, spot_csv_, sparse_spot_csv_, sparse_edge_csv_, sparse_val):
        """
        Generate sparse spots by filtering and adjusting frames based on sparse edges.

        Args:
            spot_csv_ (str): Path to the input spots CSV file.
            sparse_spot_csv_ (str): Path to save the output sparse spots CSV file.
            sparse_edge_csv_ (str): Path to the sparse edges CSV file.
            sparse_val (int): The interval value for generating sparse spots.

        Returns:
            int: 0 on successful completion.
        """
        
        
        print(f"Generating sparse spots with sparse value {sparse_val}...")
        
        # Load the sparse edges and spots data
        print("Loading sparse edges and spots data...")
        edges_df = pd.read_csv(sparse_edge_csv_)
        spots_df = pd.read_csv(spot_csv_)

        # Collect all unique spot IDs from the edges data
        spot_ids = list(edges_df['SPOT_SOURCE_ID'].values)
        spot_ids.extend(edges_df['SPOT_TARGET_ID'].values)
        spot_ids = list(set(spot_ids))

        # Filter spots DataFrame for only the relevant spot IDs
        print("Filtering relevant spot IDs...")
        filtered_spots_df = spots_df.loc[spots_df['ID'].isin(spot_ids)]

        # Update the FRAME column by applying sparse value adjustment
        print("Adjusting FRAME values...")
        filtered_spots_df.loc[:, 'FRAME'] = filtered_spots_df['FRAME'] // sparse_val

        # Save the updated spots DataFrame to a CSV file
        print(f"Saving sparse spots to: {sparse_spot_csv_}")
        filtered_spots_df.to_csv(sparse_spot_csv_, index=False)
        
        print("Sparse spots generation completed successfully.")
        return 0

    def find_unique_trkid(self, edge_csv_, utid_csv_):
        """Find unique track ID and parent ID.
        Make unique track labels start from 1.
        Make parent label 0 as default.

        Args:
            edge_csv_ (pandas DataFrame): the edge csv file to extend
            utid_csv_ (pandas DataFrame): the unique track ID csv file
        """
        
        print("Initializing unique track ID processing...")
        
        # varibles to hold unique track information
        self.lineage = dict()  # key: spot_id, value: (track_id_unique, track_id_parent)
        self.utid_max = 1  # global information for naming unique track ID

        # read edges
        print(f"Reading edge data from: {edge_csv_}")
        df = pd.read_csv(edge_csv_)
        df = df.sort_values(by=['FRAME'])

        def dfs_find_trkid(spot_id_, parent_tid_):
            """Depth-first traversal a family of tracks to assign unique track ids.

            Args:
                spot_id_ (int): spot ID
                parent_tid_ (int): parent track ID

            Raises:
                ValueError: when there are more than three edges on a spot.
            """

            self.lineage[spot_id_] = (self.utid_max, parent_tid_)
            df_edges = df[df['SPOT_SOURCE_ID']==spot_id_]

            # the track continues
            if len(df_edges) == 1:
                tgt_spot_id = df_edges['SPOT_TARGET_ID'].iloc[0]
                dfs_find_trkid(tgt_spot_id, parent_tid_=parent_tid_)
                
            elif len(df_edges) > 1:
                parent_tid = self.utid_max
                for index in range(len(df_edges)):
                    tgt_spot_id = df_edges['SPOT_TARGET_ID'].iloc[index]
                    self.utid_max += 1
                    dfs_find_trkid(tgt_spot_id, parent_tid) 
        
        # find unique track ids by a family of tracks
        print("Assigning unique track IDs...")
        for tid in df['TRACK_ID'].unique():
            # get the first frame of the track
            fid_min = df[df['TRACK_ID']==tid]['FRAME'].min()
            # get the first spot of the track
            spot_id = df[(df['TRACK_ID']==tid)&(df['FRAME']==fid_min)]['SPOT_SOURCE_ID'].iloc[0]
            # depth-first traverse and assign unique track ID
            dfs_find_trkid(spot_id, parent_tid_=0)
            self.utid_max += 1

        # convert the dictionary for creating DataFrame
        print("Converting lineage data to DataFrame...")
        data = {'spot_id': [], 'track_id_unique': [], 'track_id_parent': []}
        for sid, info in self.lineage.items():
            data['spot_id'].append(sid)
            data['track_id_unique'].append(info[0])
            data['track_id_parent'].append(info[1])
        utid_df = pd.DataFrame(data)
        utid_df.to_csv(utid_csv_, index=False)
        print(f"found {len(set(data['track_id_unique']))} tracks from {len(set(df['TRACK_ID']))} lineage.")
        print(f"saved spot ids and associated unique track ids to {utid_csv_}")

        return 0
    
    def find_mitosis_frame(self, edge_csv_, utid_csv_, mito_csv_):
        """Find mitosis frames for tracks.

        Args:
            edge_csv_ (str): Path of CSV file containing track edge information.
            utid_csv_ (str): Path of CSV file containing track ID information.
            mito_csv_ (str): Path of CSV file to save mitosis frame information.

        Raises:
            RuntimeError: The number of edges of the end spot in the track is invalid.

        Returns:
            int: 0 means no errors.
        """

        import pandas as pd
        from pathlib import Path
        
        print("Initializing mitosis frame detection...")

        df_mito = pd.DataFrame(columns=['track_id_unique', 'mitosis_frame'])

        if not Path(utid_csv_).is_file():
            self.find_unique_trkid(edge_csv_, utid_csv_)

        # Read data
        print(f"Reading data from:\n  Edge CSV: {edge_csv_}\n  Unique Track ID CSV: {utid_csv_}")
        edge_df = pd.read_csv(edge_csv_)
        utid_df = pd.read_csv(utid_csv_)

        tids = utid_df['track_id_unique'].unique()
        for tid in tids:
            trk_spots = utid_df[utid_df['track_id_unique'] == tid]['spot_id'].tolist()
            trk_edge_df = edge_df[edge_df['SPOT_SOURCE_ID'].isin(trk_spots)]
            trk_frame_max = trk_edge_df['FRAME'].max()
            num_edges = len(trk_edge_df[trk_edge_df['FRAME'] == trk_frame_max])

            # Non-mitosis track
            if num_edges == 0 or num_edges == 1:
                df_mito = pd.concat([df_mito, pd.DataFrame({'track_id_unique': [tid], 'mitosis_frame': [-1]})])
            # Mitosis track
            else:
                #print('mito',trk_edge_df[trk_edge_df['FRAME'] == trk_frame_max])
                df_mito = pd.concat([df_mito, pd.DataFrame({'track_id_unique': [tid], 'mitosis_frame': [trk_frame_max]})])
            #else:
            #   raise RuntimeError('The number of edges of an end spot should be within {1, 2}.')

        # Save the result
        df_mito = df_mito[df_mito['mitosis_frame'] != -1]  # Only keep track IDs with mitosis events
        df_mito.to_csv(mito_csv_, index=False)
        print(f"Saved the track IDs and associated mitosis frames to {mito_csv_}.")

        return 0

    def interpolate_spots(self, spots_csv_, utid_csv_, spots_interp_csv_):

        spots_df = pd.read_csv(spots_csv_)  # contain basis spot info
        utid_df = pd.read_csv(utid_csv_)  # contain unique track ids

        spots_df = spots_df[['TRACK_ID', 'FRAME', 'ID', 'POSITION_X', 'POSITION_Y']]
        spots_df = spots_df[spots_df['TRACK_ID']!=-1]  # filter out unlinked spots

        # fill unique and parent track ids into DataFrame
        spots_df = pd.merge(left=spots_df, right=utid_df, how='left', left_on='ID', right_on='spot_id')
        spots_df = spots_df.drop(['spot_id'], axis=1)
        #print(spots_df.shape)
        
        #spots_df['track_id_unique'] = -1  # default as placeholder
        #spots_df['track_id_parent'] = -1  # default as placeholder
        tids = utid_df['track_id_unique'].unique()  # track ids
        max_spot_id = spots_df['ID'].max()  # a start for naming filled spots
        
        # fill the holds for each track
        for tid in tids:
            # collect spots in a track
            trk_spots = utid_df[utid_df['track_id_unique']==tid]['spot_id'].tolist()
            # collect information of spots in a track
            trk_df = spots_df[spots_df['ID'].isin(trk_spots)]
            # get the original track id used when filling new spot
            old_tids = trk_df['TRACK_ID'].unique()
            assert len(old_tids) == 1, 'the track spots must have the same track ID.'
            old_tid = old_tids[0]

            # get the parent track ID used when filling new spot
            pids =utid_df[utid_df['track_id_unique']==tid]['track_id_parent'].unique()
            assert len(pids) == 1, 'the track spots must have the same track ID.'
            pid = pids[0]

            # check if interpolation is needed
            f_src = trk_df['FRAME'].min()  # source frame
            f_end = trk_df['FRAME'].max()  # end frame
            trk_span = f_end - f_src + 1
            num_trk_spots = len(trk_df)

            # if no holes in the track, skip the track
            if num_trk_spots == trk_span:
                continue

            # interpolate by finding gap and filling it, iteratively
            frame_ids = trk_df['FRAME'].sort_values().tolist()

            for i in range(num_trk_spots-1):
                gap = int(frame_ids[i+1] - frame_ids[i])
                # if two frames are continuous, the gap is 1
                if gap == 1:
                    continue

                # use frame ID to select end spots and positions
                src_id = trk_df[trk_df['FRAME']==frame_ids[i]]['ID'].iloc[0]
                tgt_id = trk_df[trk_df['FRAME']==frame_ids[i+1]]['ID'].iloc[0]

                x1 = trk_df[trk_df['ID']==src_id]['POSITION_X'].iloc[0]
                y1 = trk_df[trk_df['ID']==src_id]['POSITION_Y'].iloc[0]
                x2 = trk_df[trk_df['ID']==tgt_id]['POSITION_X'].iloc[0]
                y2 = trk_df[trk_df['ID']==tgt_id]['POSITION_Y'].iloc[0]

                # fill each hole in the gap
                for j in range(gap - 1):
                    pos_x = x1 + (x2 - x1) / gap * (j + 1)
                    pos_y = y1 + (y2 - y1) / gap * (j + 1)

                    # create a new spot ID
                    max_spot_id += 1
                    fill_frame_id = frame_ids[i] + j + 1
                    spot_info = {'TRACK_ID':old_tid,
                                 'FRAME':fill_frame_id,
                                 'ID':max_spot_id,
                                 'POSITION_X':pos_x,
                                 'POSITION_Y':pos_y,
                                 'track_id_unique':tid,
                                 'track_id_parent':pid}

                    # add the interpolated spot
                    #spots_df = spots_df.append(spot_info, ignore_index=True)
                    spot_info_df = pd.DataFrame([spot_info])
                    spots_df = pd.concat([spots_df, spot_info_df],ignore_index=True)

        spots_df.to_csv(spots_interp_csv_, index=False)
        return 0


    def append_velocity(self, spots_csv_, new_spots_csv_, gaps_=[1, 2], noise_std_=[0.0, 10], scale_=1):
        """Find the ground truth velocity (as dis) for given time intervals (gaps).
        The position is converted to pixel unit.
        Optionally add noises. Note the spots in the tracks should be continuous,
        which means the number of frame between two consecutive spots should be one.

        Args:
            spots_csv_ (str): file path to the spots csv file.
            new_spots_csv_ (str): file path to save the new spots csv file.
            gaps_ (list): gaps between the current spot and next spots.
            noise_std_ (list): standard deviation (pixel) of noises adding to velocity component.
            scale_ (double): "pixel unit = current unit / scale_ "

        returns:
            int: 0 if exit normally.
        """
        spots_df = pd.read_csv(spots_csv_)
        spots_df['pos_x'] = spots_df['POSITION_X'] / scale_
        spots_df['pos_y'] = spots_df['POSITION_Y'] / scale_
        tids = spots_df['track_id_unique'].unique()  # collect track ids

        # for each track, set future positions in terms of track data frame
        for tid in tids:
            trk_df = spots_df[spots_df['track_id_unique']==tid]  # spots in a track
            f_end = int(trk_df['FRAME'].max())  # end frame

            for _, row in trk_df.iterrows():
                for gap in gaps_:
                    for noise_id, std in enumerate(noise_std_):
                        f_next = row['FRAME'] + gap
                        col_dx = f"dt{gap}_n{noise_id}_dx"
                        col_dy = f"dt{gap}_n{noise_id}_dy"

                        # the spot in track after interval exists
                        if f_next <= f_end:
                            x_next = trk_df[trk_df['FRAME']==f_next]['pos_x'].iloc[0]
                            y_next = trk_df[trk_df['FRAME']==f_next]['pos_y'].iloc[0]

                            dx = x_next - row['pos_x']
                            dy = y_next - row['pos_y']

                            # add noise to position x
                            dx += np.random.normal(0, std * (gap ** 0.5))
                            # add noise to position y
                            dy += np.random.normal(0, std * (gap ** 0.5))
                        # the spot in track after interval doesn't exist, assign zero
                        else:
                            dx = 0.0
                            dy = 0.0

                        spots_df.loc[spots_df['ID']==row['ID'], col_dx] = dx
                        spots_df.loc[spots_df['ID']==row['ID'], col_dy] = dy

        spots_df = spots_df.drop(columns=['POSITION_X', 'POSITION_Y'])
        # If spots csv does not exist
        if Path(new_spots_csv_).is_file():
            # If exists, extend it.
            spots_df_old = pd.read_csv(new_spots_csv_)
            cols_to_join = ['ID']
            for gap in gaps_:
                for noise_id in range(len(noise_std_)):
                    cols_to_join.append(f"dt{gap}_n{noise_id}_dx")
                    cols_to_join.append(f"dt{gap}_n{noise_id}_dy")
            spots_df_ext = spots_df_old.merge(spots_df[cols_to_join], on='ID', how='left')
            spots_df_ext .to_csv(new_spots_csv_, index=False)

        else:
            spots_df.to_csv(new_spots_csv_, index=False)

        print(f"Extended spots with velocity in pixel and saved it to {new_spots_csv_}")

        return 0
    
    def generate_prelap(self, seq_, gap):
        """
        Generate a prelap data .

        Args:
            seq_ (str): Sequence name of the dataset.
            gap (int): Gap between frames for processing.

        Returns:
            int: 0 on successful completion.
        """
        
        print(f"Generating prelap data for sequence '{seq_}' with gap {gap}...")

        # Load the spots and unique track IDs data
        print("Loading data...")
        spots_df = pd.read_csv(get_spots_vel_csv(seq_))
        utids_df = pd.read_csv(get_utids_csv(seq_))
        prelap_path = get_prelap_save_path(seq_, gap)

        # Define the required columns
        columns_to_select = [
            'FRAME', 'ID', 'track_id_unique', 'track_id_parent', 'pos_x', 'pos_y',
            f'dt{gap}_n0_dx', f'dt{gap}_n0_dy'
        ]
        spots_df = spots_df[columns_to_select].copy()

        # Filter rows based on the frame conditions
        print("Filtering data based on frame conditions...")
        spots_df = spots_df[(spots_df['FRAME'] == 0) | (spots_df['FRAME'] % gap == 0)]

        # Initialize dictionaries and counters
        dict_lap_pred = {}
        dict_frame = {}
        lap_pred_counter = 0
        frame_counter = 0

        # Iterate through the rows of the filtered spots DataFrame
        print("Processing rows...")
        for index, row in spots_df.iterrows():
            # Map FRAME values to sequential integers
            if row['FRAME'] not in dict_frame:
                dict_frame[row['FRAME']] = frame_counter
                frame_counter += 1
            spots_df.at[index, 'FRAME'] = dict_frame[row['FRAME']]

            # Assign unique prediction track IDs
            if row['track_id_unique'] not in dict_lap_pred:
                lap_pred_counter += 1
                dict_lap_pred[row['track_id_unique']] = lap_pred_counter
                row['tid_lap_pred'] = lap_pred_counter
            else:
                row['tid_lap_pred'] = dict_lap_pred[row['track_id_unique']]
            spots_df.at[index, 'tid_lap_pred'] = row['tid_lap_pred']

            # Calculate distances and determine next ID
            dx = row[f'dt{gap}_n0_dx']
            dy = row[f'dt{gap}_n0_dy']
            

            matching_indices = utids_df[
                (utids_df['track_id_unique'] == row['track_id_unique']) & 
                (utids_df['spot_id'] == row['ID'])
            ].index

            if len(matching_indices) > 0:
                matching_index = matching_indices[0]
                if (matching_index + gap < utids_df.shape[0]) and (
                    utids_df.iloc[matching_index + gap]['track_id_unique'] == utids_df.iloc[matching_index]['track_id_unique']
                ):
                    spots_df.at[index, 'ID_next'] = utids_df.iloc[matching_index + gap]['spot_id']
                    spots_df.at[index, 'dist_link'] = math.sqrt(dx * dx + dy * dy)
                else:
                    spots_df.at[index, 'ID_next'] = 0
                    spots_df.at[index, 'dist_link'] = 'inf'
            #else:
            #    spots_df.at[index, 'ID_next'] = 0
            #    spots_df.at[index, 'dist_link'] = 'inf'

        # Ensure proper formatting of ID_next
        print("Formatting data...")
        spots_df['ID_next'] = spots_df['ID_next'].fillna(0).astype(int)

        # Save the processed DataFrame to a CSV file
        print(f"Saving prelap data to: {prelap_path}")
        spots_df.to_csv(prelap_path, index=False)
        
        print("Prelap data generation completed successfully.")
        return 0

class Sampler:
    """Make a dataset at a specific interval by down-sampling."""
    def sample(df_, gap_=1, num_frames_=289, keep_first_=True):
        """
        Note here keep_first only help tracks starting from the
        first frame to have known previous velocity.
        """
        start = 0 if keep_first_ else gap_
        fids = range(start, num_frames_, gap_)
        df_sampled = df_[df_['FRAME'].isin(fids)]
        # convert the frame id to continuous integers
        df_sampled = df_sampled.apply(lambda col: col//gap_ if col.name == 'FRAME' else col)

        return df_sampled

    def upsample(df_, gap_):
        df_upsampled = df_.apply(lambda col: col*gap_ if col.name == 'FRAME' else col)
        return df_upsampled