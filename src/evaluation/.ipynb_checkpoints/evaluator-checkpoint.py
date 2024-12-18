from pathlib import Path, PurePath
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.helper.constant import epsilon


def get_df_evaluation(in_csv_, *, out_csv_=None, overwrite_=False):
    if (not overwrite_) and (out_csv_ != None) and (Path(out_csv_).is_file()):
    #if False:
        print("Evaluation csv exists, loading it.")
        df = pd.read_csv(out_csv_)

    else:
        # Evaluation.
        evaluator = TrackEvaluator(in_csv_)
        link_measure = evaluator.get_link_measure_in_spot()
        track_fraction = evaluator.get_track_fraction()

        # Load data.
        df = pd.read_csv(in_csv_)
        df['link_type'] = -2  # initial value
        df['length_track'] = 0  # initial value
        df['length_max_rec'] = 0  # initial value
        df['fraction_max_rec'] = 0  # initial value

        # Set the link type.
        for spot, link_type in link_measure.items():
            df.loc[df['ID']==spot, 'link_type'] = link_type

        # Set length of reconstructed and correct track.
        for tid, (len_max, len_track) in track_fraction.items():
            df.loc[df['track_id_unique']==tid, 'length_max_rec'] = len_max
            df.loc[df['track_id_unique']==tid, 'length_track'] = len_track

        # Calculate the fraction of track reconstruction.
        df['fraction_max_rec'] = df['length_max_rec'] / df['length_track']

        if out_csv_ != None:
            # Save the extended dataframe
            df.to_csv(out_csv_, index=False)
            print(f"Saved evaluation result to {out_csv_}.")

    return df


class TrackEvaluator:
    def __init__(self, res_csv_=None):
        self.res_csv = res_csv_
        self.df = pd.read_csv(self.res_csv)
        self.spots = self._get_spots()
        self.ref_graph = self._get_graph('ref')
        self.est_graph = self._get_graph('est')
        self.num_track_ref = len(self.df['track_id_unique'].unique())
        self.num_track_est = len(self.df['track_id_unique_pred'].unique())
        self.track_fraction = None
        self.complete_track = None


    def _get_spots(self):
        df = self.df  # for convenience
        spots = [(row[0], {'FRAME':row[1],'pos_x':row[2],'pos_y':row[3],
                           'track_id_unique':row[4], 'track_id_parent':row[5],
                           'track_id_unique_pred':row[6],'track_id_parent_pred':row[7]})
                  for row in
                  zip(df['ID'], df['FRAME'],df['pos_x'],df['pos_y'],
                      df['track_id_unique'], df['track_id_parent'],
                      df['track_id_unique_pred'],df['track_id_parent_pred'])]
        return spots


    def _get_graph(self, type_):
        df = self.df  # for convenience
        graph = nx.DiGraph()
        graph.add_nodes_from(self.spots)  # add graph nodes

        if type_ == 'ref':
            col_suffix = ''
        elif type_ == 'est':
            col_suffix = '_pred'
        else:
            raise ValueError('Unrecognized type which should be ref or est.')

        tid_col = 'track_id_unique' + col_suffix
        pid_col = 'track_id_parent' + col_suffix

        tids = df[tid_col].unique()  # track IDs

        # Add edges and corresponding track ID and parent ID
        for tid in tids:
            df_track = df[df[tid_col]==tid]
            assert len(df_track[pid_col].unique()) == 1
            pid = df_track[pid_col].iloc[0]  # track ID of parent
            fid_start = df_track['FRAME'].min()
            fid_end = df_track['FRAME'].max()

            # Iterate the spots in track to get edges.
            for fid in range(int(fid_start), int(fid_end)):
                try:
                    # Use frame ID to locate spot ID in the track
                    spot_id_up = df_track[df_track['FRAME']==fid]['ID'].iloc[0]
                    spot_id_down = df_track[df_track['FRAME']==fid+1]['ID'].iloc[0]
                    graph.add_edge(spot_id_up, spot_id_down, track_id=tid, parent_id=pid)
                    # TODO
                    # Later, may add edges for cell division here.
                except:
                    pass

        return graph


    def get_link_measure(self):
        all_pos = len(self.ref_graph.edges)
        all_est = len(self.est_graph.edges)
        graph_intersection = nx.intersection(self.ref_graph, self.est_graph)
        true_pos = len(graph_intersection.edges)
        false_neg = all_pos - true_pos
        false_pos = all_est - true_pos

        return true_pos, false_pos, false_neg


    def get_link_measure_in_spot(self):
        """Classify spots into difference linking errors.
        Returns:
            link_measure: A dictionary (spot id: link error).
            -1: default value.
            0: true positive.
            1: false positive.
            2: false negative.
            3: both false positve and false negative.
        """
        ref_edges = {edge for edge in self.ref_graph.edges}
        est_edges = {edge for edge in self.est_graph.edges}
        tp_edges = ref_edges & est_edges  # intersection
        fp_edges = est_edges - ref_edges  # difference
        fn_edges = ref_edges - est_edges  # difference
        link_measure = {spot: -1 for spot, info in self.spots}  # default -1

        for (src_spot, tgt_spot) in tp_edges:
            link_measure[src_spot] = 0

        for (src_spot, tgt_spot) in fp_edges:
            link_measure[src_spot] = 1

        for (src_spot, tgt_spot) in fn_edges:
            # Pure false negative.
            if link_measure[src_spot] != 1:
                link_measure[src_spot] = 2
            # Both False negative and false positive.
            else:
                link_measure[src_spot] = 3

        return link_measure


    def get_precision(self):

        graph_intersection = nx.intersection(self.ref_graph, self.est_graph)
        true_pos = len(graph_intersection.edges)
        all_est = len(self.est_graph.edges)

        precision = true_pos / (all_est + epsilon)

        return precision


    def get_recall(self):

        graph_intersection = nx.intersection(self.ref_graph, self.est_graph)
        true_pos = len(graph_intersection.edges)
        all_pos = len(self.ref_graph.edges)

        recall = true_pos / (all_pos + epsilon)

        return recall


    def get_track_fraction(self):
        raw_track_fraction = dict()

        df = self.df  # for naming convenience

        tids = df['track_id_unique'].unique()  # track IDs

        # Add edges and corresponding track ID and parent ID
        for tid in tids:
            df_track = df[df['track_id_unique']==tid]
            assert len(df_track['track_id_unique'].unique()) == 1

            len_track = len(df_track)  # length of track

            # When the track incudes only one cell.
            if len_track == 1:
                matched_tid_pred = df[df['track_id_unique']==tid]['track_id_unique_pred'].iloc[0]
                len_max = 1
                raw_track_fraction[tid] = (len_max, len_track, matched_tid_pred)

            # When the track incudes multiple cells.
            else:
                fid_start = df_track['FRAME'].min()
                fid_end = df_track['FRAME'].max()

                fid_root = fid_start  # frame of the root node of current estimated track
                fid_leaf = fid_start  # frame of the leaf node of current estimated track

                # Iterate the spots in track to check edges.
                # Criteria of continuous track:
                # 1. Reference track edge is in the estimated track edges.
                # 2. The consecutive edges in reference track has the same track ID
                #    in the estimated tracks.
                len_max = 1  # max length of sub track
                tid_predecessor = -1  # denotes no predecessor link and track
                matched_tid_pred = -1

                while fid_leaf < fid_end:
                    # Use frame ID to locate spot ID in the track.
                    spot_id_up = df_track[df_track['FRAME']==fid_leaf]['ID'].iloc[0]
                    spot_id_down = df_track[df_track['FRAME']==fid_leaf+1]['ID'].iloc[0]

                    # Check if current edge is correctly detected.
                    if (spot_id_up, spot_id_down) in self.est_graph.edges:
                        # Check if the current edge is correctly assigned to track.
                        tid_current = self.est_graph[spot_id_up][spot_id_down]['track_id']

                        # This is the first edge checked or new sub track.
                        if tid_predecessor==-1 or tid_current == tid_predecessor:
                            fid_leaf += 1  # Update the leaf of the estimated track.
                            # Update max sub track length if needed.
                            if len_max < fid_leaf - fid_root + 1:
                                len_max = fid_leaf - fid_root + 1
                                matched_tid_pred = tid_current

                        # Update the predecessor track ID
                        tid_predecessor = tid_current

                    else:
                        # Update the endpoints of ongoing sub track.
                        fid_root = fid_leaf + 1
                        fid_leaf = fid_root

                        # Update the predecessor track ID
                        tid_predecessor = -1

                raw_track_fraction[tid] = (len_max, len_track, matched_tid_pred)

        # Refine the track fraction dictionary by removing boubled assigned
        # predicted track ID.
        used_tid_pred = set()
        track_fraction = dict()

        for tid, (len_max, len_track, matched_tid_pred) in raw_track_fraction.items():
            if matched_tid_pred not in used_tid_pred:
                track_fraction[tid] = (len_max, len_track)
                used_tid_pred.add(matched_tid_pred)
            # the predicted track is assigned multiple times
            else:
                track_fraction[tid] = (0, len_track)

        return track_fraction


    def set_track_fraction(self, track_fraction_):
        self.track_fraction = track_fraction_


    def get_avg_max_fraction(self):

        track_fraction = self.get_track_fraction()
        max_fraction = [len_max / len_track for (len_max, len_track) in track_fraction.values()]
        avg_max_fraction = sum(max_fraction) / len(max_fraction)

        return avg_max_fraction


    def get_complete_track(self):

        track_fraction = self.get_track_fraction()
        complete_track = {tid for tid, info in track_fraction.items() if info[0] == info[1]}

        return complete_track


    def set_complete_track(self, complete_track_):
        self.complete_track = complete_track_


    def get_num_complete_track(self):
        complete_track = self.get_complete_track()
        num_complete_track = len(complete_track)

        return num_complete_track


    def get_num_track_ref(self):
        return self.num_track_ref


    def get_num_track_est(self):
        return self.num_track_est

    def get_ct_score(self, ctc_=True):
        # Get complete track score.
        num_complete_track = self.get_num_complete_track()
        num_track_ref = self.get_num_track_ref()
        num_track_est = self.get_num_track_est()
        score_complete_track = num_complete_track / num_track_ref
        score_complete_track_ctc = 2 * num_complete_track / (num_track_ref + num_track_est)

        # Follow the Cell Tracking Challenge definition
        if ctc_:
            score = score_complete_track_ctc
        else:
            score = score_complete_track

        return score

    def get_perf(self):
        """Calculate and return all performance."""
        perf = dict()

        tp, fp, fn = self.get_link_measure()
        perf['TP'] = tp
        perf['FP'] = fp
        perf['FN'] = fn
        perf['num_link'] = tp + fn
        perf['recall'] = self.get_recall()
        perf['precision'] = self.get_precision()
        perf['avg_max_fraction'] = self.get_avg_max_fraction()
        perf['num_complete_track'] = self.get_num_complete_track()
        perf['num_track_ref'] = self.get_num_track_ref()
        perf['num_track_est'] = self.get_num_track_est()
        perf['score_complete_track_ctc'] = self.get_ct_score()

        return perf

    def draw_graph(self, save_dir_):
        Path(save_dir_).mkdir(exist_ok=True)
        nx.draw(self.ref_graph, with_labels=True, pos=nx.planar_layout(self.ref_graph))
        edge_label =nx.get_edge_attributes(self.ref_graph,'track_id')
        nx.draw_networkx_edge_labels(self.ref_graph, pos=nx.planar_layout(self.ref_graph), edge_labels=edge_label)
        plt.savefig(str(Path(save_dir_) / 'ref_graph.png'))
        plt.close()

        nx.draw(self.est_graph, with_labels=True, pos=nx.planar_layout(self.est_graph))
        edge_label =nx.get_edge_attributes(self.est_graph,'track_id')
        nx.draw_networkx_edge_labels(self.ref_graph, pos=nx.planar_layout(self.est_graph), edge_labels=edge_label)
        plt.savefig(str(Path(save_dir_) / 'est_graph.png'))
        plt.close()