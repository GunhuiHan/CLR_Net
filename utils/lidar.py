"""
This is for preprocessing the radar. e.g. normalization

The std_map and mean_map has been calculated by 
using all radar data from nuScenes
"""

# 3rd Party Libraries
import numpy as np


def enrich_lidar_data(lidar_data):
    """
    Lidar Format: #added
        - Shape: 4 x n
        - Semantics: 
            [0]: x (1)
            [1]: y (2)
            [2]: z (3)
            [3]: intensity (4)
    """
    assert lidar_data.shape[0] == 4, "lidar Channel count mismatch."

    # Adding distance
    # Calculate distance
    dist = np.sqrt(lidar_data[0,:]**2 + lidar_data[1,:]**2)
    dist = np.expand_dims(dist, axis=0)

    intensity = np.expand_dims(lidar_data[3], axis=0)

    data_collections = [
        lidar_data,
        dist,
        intensity,
    ]

    enriched_lidar_data = np.concatenate(data_collections, axis=0)

    return enriched_lidar_data
