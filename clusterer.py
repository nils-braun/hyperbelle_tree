from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

layer_id = 0
phi_id = 1
x_id = 2
y_id = 3
number_id = 4


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

        self.layers = list(range(9))

        self.hit_masks_grouped_by_layer = {}
        self.not_used_mask = None

    def fit(self, X, y):
        pass

    @staticmethod
    @np.vectorize
    def get_weight(hit_1, hit_2):
        #return hit_1.cluster_id == hit_2.cluster_id
        return abs(hit_1 - hit_2) < 0.01

    def get_quality(self, track):
        return 1

    def walk(self, hit, track, X_event):
        # abort criteria
        if hit[layer_id] == 0:
            yield track
            return

        unused_hits_on_next_layer_mask = self.hit_masks_grouped_by_layer[hit[layer_id] - 1] & self.not_used_mask

        unused_hits_on_next_layer = X_event[unused_hits_on_next_layer_mask]

        if np.all(~unused_hits_on_next_layer_mask):
            yield track
            return

        weights = self.get_weight(hit[phi_id], unused_hits_on_next_layer[:, phi_id])

        maximal_weight = np.max(weights)

        if maximal_weight < self.cut:
            yield track
            return

        possible_next_hits_mask = maximal_weight - weights < self.duplicate_cut

        for possible_next_hit in unused_hits_on_next_layer[possible_next_hits_mask]:
            for track_candidate in self.walk(possible_next_hit,
                                             track + [int(possible_next_hit[number_id])],
                                             X_event,):
                yield track_candidate

    def predict_single_event(self, X_event):
        # Attention! We are redefining the iphi column here, as we do not need it
        X_event[:, phi_id] = np.arctan2(X_event[:, y_id], X_event[:, x_id])

        # Add a another column to store the hit ids
        number_of_hits = len(X_event)
        X_event = np.concatenate([X_event, np.reshape(np.arange(number_of_hits), (number_of_hits, 1))], axis=1)

        for layer in self.layers:
            self.hit_masks_grouped_by_layer[layer] = X_event[:, layer_id] == layer

        self.not_used_mask = np.ones(len(X_event)).astype("bool")
        labels = -1 * np.ones(len(X_event))

        track_counter = 0

        for layer in reversed(self.layers):
            while True:
                unused_mask_in_this_layer = self.hit_masks_grouped_by_layer[layer] & self.not_used_mask

                if not np.any(unused_mask_in_this_layer):
                    break

                start_hit = X_event[unused_mask_in_this_layer][0]

                track_list = list(self.walk(start_hit, [int(start_hit[number_id])], X_event))

                best_track = max(track_list, key=lambda track: self.get_quality(track))

                # Store best track
                self.not_used_mask[best_track] = False
                labels[best_track] = track_counter

                track_counter += 1

        return labels

