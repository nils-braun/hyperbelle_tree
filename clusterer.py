import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Clusterer(BaseEstimator):
    def __init__(self, cut=1, duplicate_cut=1):
        """
        Track Pattern Recognition based on the connections between two nearest hits from two nearest detector layers.
        Parameters
        ----------
        min_cos_value : float
            Minimum cos value between two nearest segments of the track.
        """
        self.cut = cut
        self.duplicate_cut = duplicate_cut

    def fit(self, X, y):
        pass

    def get_weight(self, hit_1, hit_2, X_event, track):
        #return hit_1.cluster_id == hit_2.cluster_id
        return abs(hit_1.phi - hit_2.phi) < 0.01

    def get_quality(self, track):
        return 1

    def walk(self, hit, X_event, track):
        if hit.layer == 0:
            yield track
        else:
            hits_on_next_layer = X_event[(X_event.layer == hit.layer - 1) & (X_event.not_used == True)]

            if len(hits_on_next_layer) == 0:
                yield track
            else:

                hits_on_next_layer["weight"] = hits_on_next_layer.apply(lambda next_hit: self.get_weight(hit, next_hit, X_event, track), axis=1)

                maximal_weight = hits_on_next_layer.weight.max()

                if maximal_weight < self.cut:
                    yield track
                # possible abort here
                else:
                    possible_next_hits = hits_on_next_layer[maximal_weight - hits_on_next_layer.weight < self.duplicate_cut]

                    for possible_next_hit_id in possible_next_hits.index:
                        possible_next_hit = possible_next_hits.loc[possible_next_hit_id]
                        for track_candidate in self.walk(possible_next_hit, X_event,
                                                         track + [possible_next_hit]):
                            yield track_candidate

    def predict_single_event(self, X_event):
        tmp = pd.DataFrame()

        tmp["x"] = X_event[:, 2]
        tmp["y"] = X_event[:, 3]
        tmp["layer"] = X_event[:, 0]
        tmp["phi"] = np.arctan2(tmp.y, tmp.x)

        tmp["not_used"] = True
        tmp["label"] = -1

        track_counter = 0

        while tmp.not_used.sum() > 0:
            start_hit = tmp[tmp.not_used].sort_values("layer", ascending=False).iloc[0]
            track_list = self.walk(start_hit, tmp, [start_hit])

            best_track = max(track_list, key=lambda track: self.get_quality(track))

            for hit in best_track:
                tmp.loc[hit.name, "label"] = track_counter
                tmp.loc[hit.name, "not_used"] = False
            track_counter += 1

        return tmp["label"].values

