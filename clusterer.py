from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from scipy import optimize

layer_id = 0
phi_id = 1
x_id = 2
y_id = 3
number_id = 4


class Clusterer(BaseEstimator):
    def __init__(self, cut=0.0001, duplicate_cut=0.15):
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
        
        self.weights = np.zeros((8, 50))
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
                dphi = self.abs_phi_dist(hits_phi[sort][1:], hits_phi[sort][:-1])[use]
                dphi_bin = np.round(dphi / self.dphiRange * self.weights.shape[1])
                in_range = dphi_bin < self.weights.shape[1]
                self.weights[hits_layer_end[in_range].astype('int'), dphi_bin[in_range].astype('int')] += 1
        for layer in range(self.weights.shape[0]):
            #self.weights[layer] /= np.sum(self.weights[layer])
            self.weights[layer] /= np.max(self.weights[layer])

    @staticmethod
    def fit_track(track_list, X_event):

        if len(track_list) < 3:
            return 0, 0, 0

        x_values = X_event[track_list, x_id]
        y_values = X_event[track_list, y_id]

        x_m = np.mean(x_values)
        y_m = np.mean(y_values)

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x_values - xc) ** 2 + (y_values - yc) ** 2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2 = calc_R(*center_2)
        R_2 = Ri_2.mean()
        residu_2 = sum((Ri_2 - R_2) ** 2)

        phi_values = np.arctan2(y_values - yc_2, x_values - xc_2)
        phi_2 = (np.arctan2(yc_2, xc_2) + np.sign(np.mean(phi_values)) * np.pi)

        return R_2, phi_2, residu_2

    @staticmethod
    def abs_phi_dist(phi_1, phi_2):
        dphi = abs(phi_1 - phi_2)
        dphi = (dphi > np.pi) * (2 * np.pi - dphi) + (dphi <= np.pi) * dphi

        return dphi

    @staticmethod
    def get_weight(phi_1, phi_2, dphiRange, weights):
        dphi = Clusterer.abs_phi_dist(phi_1, phi_2)
        dphi_bin = np.round(dphi / dphiRange * len(weights))
        in_range = dphi_bin < len(weights)
        w = -np.ones(len(phi_2))
        w[in_range] = weights[dphi_bin[in_range].astype('int')]
        return w

    def get_quality(self, track, X_event):
        R, phi, residuum = self.fit_track(track, X_event)

        return -residuum

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

        possible_next_hits_mask = maximal_weight - weights  < self.duplicate_cut

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
            for unfinished_track in unfinished_tracks:
                other_unfinished_tracks = unfinished_tracks_begin[last_layer + 2]

                if len(other_unfinished_tracks) == 0:
                    continue
                best_unfinished_track = max(other_unfinished_tracks, key=lambda track: self.get_quality(unfinished_track + track, X_event))
                unfinished_track += best_unfinished_track

                del other_unfinished_tracks[other_unfinished_tracks.index(best_unfinished_track)]

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

                if len(found_track_list) != 1:
                    best_track = max(found_track_list, key=lambda track: self.get_quality(track, X_event))
                else:
                    best_track = found_track_list[0]

                # Store best track
                self.not_used_mask[best_track] = False

                track_list.append(best_track)

        self.extrapolate(track_list, X_event)

        for i, track in enumerate(track_list):
            labels[track] = i

        return labels
