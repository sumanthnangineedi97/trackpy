def select_sid_left_to_link(df_, fid_, *, sep_slow_=True):
    """Get the spot (cell) IDs on the left side to link.
       Optionally skip slow cells identified.
    """
    sids_left_link = []
    spot_ids = df_[df_['FRAME']==fid_]['ID'].tolist()
    sids_left_slow = []  # slow cells identified
    print('sep_slow_',sep_slow_)
    for spot_id in spot_ids:
        # Check if the end spot using groundtruth track ID
        gt_tid = df_[df_['ID']==spot_id]['track_id_unique'].iloc[0]
        end_fid = df_[df_['track_id_unique']==gt_tid]['FRAME'].max()
        if fid_ == end_fid:
            continue

        # Check if the single-spot track
        track_len = df_[df_['track_id_unique']==gt_tid]['FRAME'].size
        if track_len == 1:
            continue

        if sep_slow_:
            # Check if identified as slow cells (when u_flag is 0)
            flag = df_.loc[df_['ID']==spot_id, 'u_flag'].iloc[0]
            if flag == 0:
                sids_left_slow.append(spot_id)  # slow cells identified
                continue

        sids_left_link.append(spot_id)

    if sep_slow_:
        print(f"Pre-linked IDs left: {sids_left_slow}.")

    return sids_left_link


def select_sid_right(df_, fid_right_):
    """Select spot IDs on the right side for linking and track assingment."""
    sids_right_link = []
    sids_right_single = []
    sids_right_slow = []

    fid_right = fid_right_
    sids_right = df_[df_['FRAME']==fid_right]['ID'].tolist()

    for spot_id in sids_right:
        gt_tid = df_[df_['ID']==spot_id]['track_id_unique'].iloc[0]

        # Check if the single-spot track.
        track_len = df_[df_['track_id_unique']==gt_tid]['FRAME'].size
        start_fid = df_[df_['track_id_unique']==gt_tid]['FRAME'].min()

        if track_len == 1:
            sids_right_single.append(spot_id)
            continue

        # Check if the start spot using groundtruth track ID.
        elif fid_right == start_fid:
            continue

        # Check if the pre-linked spots.
        elif df_.loc[df_['ID']==spot_id, 'track_id_unique_pred'].iloc[0] != '':
            sids_right_slow.append(spot_id)
            continue

        # Add qualified spot to candidate list.
        else:
            sids_right_link.append(spot_id)

    print(f"Pre-linked IDs right: {sids_right_slow}.")

    return sids_right_link, sids_right_single


def select_sid_right_to_link(df_, fid_right_):
    """Select spot IDs on the right side for linking and track assingment."""
    sids_right_link = []

    fid_right = fid_right_
    sids_right = df_[df_['FRAME']==fid_right]['ID'].tolist()

    for spot_id in sids_right:
        gt_tid = df_[df_['ID']==spot_id]['track_id_unique'].iloc[0]
        # Check if the single-spot track.
        track_len = df_[df_['track_id_unique']==gt_tid]['FRAME'].size
        start_fid = df_[df_['track_id_unique']==gt_tid]['FRAME'].min()

        if track_len == 1:
            continue
        # Check if the start spot using groundtruth track ID.
        elif fid_right == start_fid:
            continue
        # Add qualified spot to candidate list.
        else:
            sids_right_link.append(spot_id)

    return sids_right_link