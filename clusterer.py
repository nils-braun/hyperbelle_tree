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
    def __init__(self, cut=-0.1, duplicate_cut=0.05):
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
        return -abs(hit_1 - hit_2)

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

        possible_next_hits_mask = maximal_weight == weights  # < self.duplicate_cut

        for possible_next_hit in unused_hits_on_next_layer[possible_next_hits_mask]:
            for track_candidate in self.walk(possible_next_hit,
                                             track + [int(possible_next_hit[number_id])],
                                             X_event):
                yield track_candidate

    def extrapolate(self, track_list, X_event):
        unfinished_tracks_end = defaultdict(list)
        unfinished_tracks_begin = defaultdict(list)

        for track in track_list:
            if len(track) == len(self.layers):
                continue

            hits_of_track = X_event[track]
            last_layer = max(hits_of_track[:, layer_id])
            first_layer = min(hits_of_track[:, layer_id])

            if last_layer != len(self.layers) - 1:
                unfinished_tracks_end[last_layer].append(track)
            elif first_layer != 0:
                unfinished_tracks_begin[first_layer].append(track)

        for last_layer, unfinished_tracks in unfinished_tracks_end.items():
            if len(unfinished_tracks) == 1:
                unfinished_track = unfinished_tracks[0]
                other_unfinished_tracks = unfinished_tracks_begin[last_layer + 2]

                if len(other_unfinished_tracks) == 1:
                    other_unfinished_track = other_unfinished_tracks[0]
                    unfinished_track += other_unfinished_track

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

        track_list = []

        for layer in reversed(self.layers):
            while True:
                unused_mask_in_this_layer = self.hit_masks_grouped_by_layer[layer] & self.not_used_mask

                if not np.any(unused_mask_in_this_layer):
                    break

                start_hit = X_event[unused_mask_in_this_layer][0]

                found_track_list = list(self.walk(start_hit, [int(start_hit[number_id])], X_event))

                best_track = max(found_track_list, key=lambda track: self.get_quality(track))

                # Store best track
                self.not_used_mask[best_track] = False

                track_list.append(best_track)

        self.extrapolate(track_list, X_event)

        for i, track in enumerate(track_list):
            labels[track] = i

        return labels
