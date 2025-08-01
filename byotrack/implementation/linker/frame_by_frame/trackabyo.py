import dataclasses
from typing import List, Optional, Tuple, Dict, Union, Collection, Sequence
import warnings
import logging

import numpy as np
import torch
import dask.array as da
from tqdm import tqdm


import byotrack
from byotrack.implementation.linker.frame_by_frame.base import (
    AssociationMethod,
)
from byotrack.implementation.linker.frame_by_frame.nearest_neighbor import (
    NearestNeighborParameters,
    NearestNeighborLinker,
)


from trackastra.model import Trackastra
from trackastra.utils import normalize
from trackastra.data import build_windows, get_features
from trackastra.model.predict import predict_windows


def dict_builder(nodes: List[Dict], weights: List[Tuple]):
    """
    Build the dictionnary matching with the Trackastra's graph

    Args :
        nodes : The list of the graph's nodes (detections),
            shape of node {'id': ,'coords': ,'time': ,'label': }
        weights : The list of all the graph's edges with their weight
            shape of weight ((node1_id,node2_id),weight)

    Returns :
        Dictionnary of possible links between detections,
            shape of link {(detection1_time,detection1_id_on_frame,detection2_time,detection2_id_on_frame)
    """
    dico = {}
    node_by_id = {node["id"]: node for node in nodes}
    for (id1, id2), weight in weights:
        node1 = node_by_id[id1]
        node2 = node_by_id[id2]
        dico[(node1["time"], node1["label"] - 1, node2["time"], node2["label"] - 1)] = (
            -np.log(weight)
        )
    return dico


class NewTrackastra(Trackastra):
    """ "
    Subclass of Trackastra just to modify the _predict function to have delta_t=3
    """

    def __init__(self, transformer, train_args, delta_t=3, device=None):
        super().__init__(transformer, train_args, device)
        self.delta_t = delta_t
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _predict(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
    ):
        self.logger.info("Predicting weights for candidate graph")
        if normalize_imgs:
            if isinstance(imgs, da.Array):
                imgs = imgs.map_blocks(normalize)
            else:
                imgs = normalize(imgs)

        self.transformer.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        self.logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
        )

        self.logger.info("Predicting windows")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
            delta_t=self.delta_t,
        )

        return predictions


@dataclasses.dataclass
class TrackaByoParameters(NearestNeighborParameters):
    """Parameters of TrackaByoLinker



    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
        n_valid (int): Number associated detections required to validate the track after its creation.
            Default: 3
        n_gap (int): Number of consecutive frames without association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, [SPARSE_]OPT_HARD, [SPARSE_]OPT_SMOOTH)
            Default: OPT_SMOOTH
        anisotropy (Tuple[float, float, float]): Anisotropy of images (Ratio of the pixel sizes
            for each axis, depth first). This will be used to scale distances.
            Default: (1., 1., 1.)
        fill_gap (bool): Fill the gap of missed detections using a forward optical flow
            propagation (Only when optical flow is provided). We advise to rather use a
            ForwardBackward interpolation using the same optical flow: it will produce
            smoother interpolations.
            Default: False
        ema (float): Optional exponential moving average to reduce detection noise. Detection positions are smoothed
            using this EMA. Should be smaller than 1. It use: x_{t+1} = ema x_{t} + (1 - ema) det(t)
            As motion is not modeled, EMA may introduce lag that will hinder tracking. It is more effective with
            optical flow to compensate motions, in this case, a typical value is 0.5, to average the previous position
            with the current measured one. For more advanced modelisation, see `KalmanLinker`.
            Default: 0.0 (No EMA)
        split_factor (float): Allow splitting of tracks, using a second association step.
            The association threshold in this case is `split_factor * association_threshold`.
            Default: 0.0 (No splits)
        merge_factor (float): Allow merging of tracks, using a second association step.
            The association threshold in this case is `merge_factor * association_threshold`.
            Default: 0.0 (No merges)

    """

    def __init__(
        self,
        association_threshold: float = 0.69,  # A définir
        *,
        n_valid=3,
        n_gap=3,
        association_method: Union[
            str, AssociationMethod
        ] = AssociationMethod.OPT_SMOOTH,
        anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ema=1.0,
        fill_gap=False,
        split_factor: float = 0.0,
        merge_factor: float = 0.0,
    ):
        super().__init__(  # pylint: disable=duplicate-code
            association_threshold=association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            anisotropy=anisotropy,
            split_factor=1 if split_factor > 0 else 0,
            merge_factor=merge_factor,
        )


class TrackaByoLinker(NearestNeighborLinker):
    """Frame by frame linker using Trackastra associating costs.

    See `TrackaByoParamaters` for the other attributes.

    Attributes:
        specs (TrackaByoParameters): Parameters specifications of the algorithm.
            See `TrackaByoParameters`.
        model (NewTrackastra) : Model of Trackastra used to compute the association costs.
        cost_dict (Dict) : Dictionnary of all the possible link between detections associated with its cost.

    """

    progress_bar_description = "TrackaByo linking"

    def __init__(
        self,
        specs: TrackaByoParameters,
        model: NewTrackastra,
        # detections: List[
        #     Dict
        # ],  # List of detections from Trackastra (nodes) used to with the weights to make links
        # weights: List[Tuple],  # List of the weights of assocations between nodes
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)
        self.specs: TrackaByoParameters
        self.model = model
        self.cost_dict: dict[Tuple, float]

        if self.specs.fill_gap and not self.optflow:
            warnings.warn("Optical flow has not been provided. Gap cannot be filled")

    def run(
        self,
        video: Union[Sequence[np.ndarray], np.ndarray],
        detections_sequence: Sequence[byotrack.Detections],
    ) -> Collection[byotrack.Track]:
        if len(video) != len(detections_sequence):
            warnings.warn(
                f"""Expected to have one Detections for each frame of the video.

            There are {len(detections_sequence)} Detections for {len(video)} frames.
            This can lead to unexpected behavior. By default we assume that the first Detections
            is aligned with the first frame and stop when the end of shortest sequence is reached.
            """
            )

        if len(video) == 0:
            return []

        self.reset(video[0].ndim - 1)

        # Convert Byotrack datas for Trackastra

        vid = np.stack([frame.squeeze() for frame in video], axis=0)
        masks = []
        for detection in detections_sequence:
            mask = detection.segmentation.cpu().numpy()
            masks.append(mask)
        masks = np.stack(masks)

        # Then compute the cost
        predictions = self.model._predict(vid, masks)
        nodes = predictions["nodes"]
        weights = predictions["weights"]
        self.cost_dict = dict_builder(nodes, weights)

        progress_bar = tqdm(
            desc=self.progress_bar_description,
            total=min(len(video), len(detections_sequence)),
        )

        for frame, detections in zip(
            [np.zeros((1, 1, 1, 1)) for _ in range(len(detections_sequence))],
            detections_sequence,
        ):
            self.update(frame, detections)
            progress_bar.update()

        progress_bar.close()

        tracks = self.collect()

        # Check produced tracks
        byotrack.Track.check_tracks(tracks, warn=True)
        return tracks

    def cost(
        self, frame: np.ndarray, detections: byotrack.Detections
    ) -> Tuple[torch.tensor, float]:
        """Compute the association cost between active tracks and detections
        
        Args:
            frame_id (int): The index of thecurrent frame of the video


        Returns:
            torch.Tensor: The cost matrix between active tracks and detections
                Shape: (n_tracks, n_dets), dtype: float
            float: The association threshold to use.

        """
        nb_lignes = len(self.active_tracks)
        nb_colonnes = detections.length
        cost = torch.full((nb_lignes, nb_colonnes), torch.inf)
        for index, track in enumerate(self.active_tracks):
            for index_det in range(nb_colonnes):
                for i in range(1, self.specs.n_gap + 2):
                    try:
                        edge = (
                            self.frame_id - i,
                            track.detection_ids[-i],
                            self.frame_id,
                            index_det,
                        )
                        if edge in self.cost_dict:
                            cost[index, index_det] = float(self.cost_dict[edge])
                            break
                    except IndexError:
                        break
        return (cost, self.specs.association_threshold)
