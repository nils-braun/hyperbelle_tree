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
    def __init__(self, cut=0.0001, duplicate_cut=1):
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
        
        self.weights = np.zeros((8, 100))
        self.dphiRange = 0.1

    def fit(self, X, y):
        events = y[:, 0]
        for event in np.unique(events):
            event_indices = events == event
            X_event = X[event_indices]
            pixelx = X_event[:, 3]
            pixely = X_event[:, 4]
            phi = np.arctan2(pixely, pixelx)
            layer = X_event[:, 1]
            particles = y[event_indices][:, 1]
            for particle in np.unique(particles):
                hits_particle = particles == particle
                hits_layer = layer[hits_particle]
                hits_phi = phi[hits_particle]
                sort = np.argsort(hits_layer)
                dlayer = hits_layer[sort][1:] - hits_layer[sort][:-1]
                use = dlayer == 1
                hits_layer_end = hits_layer[sort][:-1][use]
                dphi = np.abs(hits_phi[sort][1:] - hits_phi[sort][:-1])
                dphi = np.minimum(dphi[use], 2. * np.pi - dphi[use])
                dphi_bin = np.round(dphi / self.dphiRange * self.weights.shape[1])
                in_range = dphi_bin < self.weights.shape[1]
                self.weights[hits_layer_end[in_range].astype('int'), dphi_bin[in_range].astype('int')] += 1
        for layer in range(self.weights.shape[0]):
            #self.weights[layer] /= np.sum(self.weights[layer])
            self.weights[layer] /= np.max(self.weights[layer])

    @staticmethod
    #@np.vectorize
    def get_weight(phi_1, phi_2, dphiRange, weights):
        #return hit_1.cluster_id == hit_2.cluster_id
        #return -abs(hit_1 - hit_2)
        dphi = np.abs(phi_1 - phi_2)
        dphi = np.minimum(dphi, 2. * np.pi - dphi)
        dphi_bin = np.round(dphi / dphiRange * len(weights))
        in_range = dphi_bin < len(weights)
        w = -np.ones(len(phi_2))
        w[in_range] = weights[dphi_bin[in_range].astype('int')]
        return w

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

        weights = self.get_weight(hit[phi_id], unused_hits_on_next_layer[:, phi_id], self.dphiRange, self.weights[int(hit[layer_id] - 1)])

        maximal_weight = np.max(weights)

        if maximal_weight < self.cut:
            yield track
            return

        possible_next_hits_mask = weights == maximal_weight

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

