"""
Trackpy Tracker.

"""
from scipy.optimize import linear_sum_assignment
from src.tracking.df_filter import select_sid_left_to_link, select_sid_right
from src.tracking.trackpy_linking import TrackpyLinking
import trackpy as tp


class TrackpyTracker():
    def __init__(self):
        self.track_count = 0
        self.linking = TrackpyLinking()

    def _assign_new_track_id(self, df_, fid_):
        """Assign new track IDs for the given frame."""
        # Assign empty IDs by new ID (empty because of being start spots)
        # Frame ID -> Spot IDs -> Spots IDs with empty track ID -> incrementally assign
        spot_ids = df_[df_['FRAME']==fid_]['ID'].tolist()
        for spot_id in spot_ids:
            if df_[df_['ID']==spot_id]['track_id_unique_pred'].iloc[0] == '':
                self.track_count += 1
                df_.loc[df_['ID']==spot_id, 'track_id_unique_pred'] = self.track_count

        return df_

    def _assign_slow_track_id(self, df_, fid_left_):
        """Assign predicted track ID to spots pre-linked by slow cell classifier."""
        # Collect slow cells on the left side.
        df_slow = df_[(df_['FRAME']==fid_left_)&(df_['u_flag']==0)]
        # Assign track ID to slow cells on the right side.
        for _, row in df_slow.iterrows():
            # Check to avoid setting track ID to new tracks, maybe by mitosis.
            fid_trk_max = max(df_[df_['track_id_unique']==row['track_id_unique']]['FRAME'])
            if fid_left_ == fid_trk_max:
                print(f"Detected a slow cell at the track end at frame {fid_left_}.")
                continue
            tid = row['track_id_unique_pred']
            sid_next = row['ID_next']
            df_.loc[df_['ID']==sid_next, 'track_id_unique_pred'] = tid

        return df_

    def _assign_fast_track_id(self, df_, cols_linked_, sids_left_, sids_right_):
        for row, col in enumerate(cols_linked_):
            row_tid = df_[df_['ID']==sids_left_[row]]['track_id_unique_pred'].iloc[0]
            df_.loc[df_['ID']==sids_right_[col], 'track_id_unique_pred'] = row_tid

        return df_

    def _assign_right_single_track_id(self, df_, sids_):
        """
        Assign new track ID to single-cell track on the right side.
        """
        for spot_id in sids_:
            self.track_count += 1
            df_.loc[df_['ID']==spot_id, 'track_id_unique_pred'] = self.track_count

        return df_

    def link_df(self, df_, fcc_=False):
        """Assign track IDs to each cell in the dataframe."""
        df = df_
        fcc = fcc_  # whether to use fast cell classifier
        search_range = 1000
        # Initialization.
        self.track_count = 0  # count tracks for assigning ID
        df.insert(len(df.columns), 'track_id_unique_pred', '')
        df.insert(len(df.columns), 'track_id_parent_pred', 0)
        df.insert(len(df.columns), 'predicted', 0)  # prediction status
        
        # Iterate pairs of frames.
        frame_ids = sorted(df['FRAME'].unique())

        for i_fid, fid in enumerate(frame_ids[:-1]):
            print(f"Linking frame {i_fid}, frame ID {fid}.")
            # Step 1: Assign track ID for left side, including single-spot tracks.
            df = self._assign_new_track_id(df, fid)

            # Step 2: Collect qualified spots (start & middle) on the left side for linking.
            sids_left_link = select_sid_left_to_link(df, fid, sep_slow_=fcc)

            # Step 3: Pre-linking
            if fcc:
                df = self._assign_slow_track_id(df, fid)

            # Step 4: Collect qualified spots (end & middle) on the right side for linking.
            sids_right_link, sids_right_single = select_sid_right(df, fid+1)

            # Step 5: Linking by Hungarian algorithm following Bayes' rule.
            if len(sids_left_link) != 0 and len(sids_right_link) != 0:
                
                rows_linked, cols_linked = self.linking.normal_linking(df, sids_left_link, sids_right_link, search_range)
                
                # Step 6.1: Assign track ID for right side, including end & middle spots
                df = self._assign_fast_track_id(df, cols_linked, sids_left_link, sids_right_link)

            # Step 6.2: Assign track ID for right-side single-spot tracks.
            df = self._assign_right_single_track_id(df, sids_right_single)

        return df