
import numpy as np
import geomstats.backend as gs

# from geomstats.geometry.special_euclidean import SpecialEuclidean
# from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric

# import matplotlib.pyplot as plt
# import geomstats.visualization as visualization

# import geomstats.datasets.utils as data_utils

# import matplotlib.image as mpimg

import os



import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
)

SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")
N_STEPS = 40


def main():
    """Plot geodesics on SE(2) with different structures."""
    theta = gs.pi / 6
    initial_tangent_vec = gs.array(
        [[0.0, -theta, 0.5], [theta, 0.0, 0.5], [0.0, 0.0, 0.0]]
    )
    t = gs.linspace(-2.0, 2.0, N_STEPS + 1)
    tangent_vec = gs.einsum("t,ij->tij", t, initial_tangent_vec)
    group_geo_points = SE2_GROUP.exp(tangent_vec)
    left_geo_points = LEFT_METRIC.exp(tangent_vec)
    right_geo_points = RIGHT_METRIC.exp(tangent_vec)

    ax = visualization.plot(
        group_geo_points, space="SE2_GROUP", color="black", label="Group"
    )
    ax = visualization.plot(
        left_geo_points, ax=ax, space="SE2_GROUP", color="yellow", label="Left"
    )
    ax = visualization.plot(
        right_geo_points,
        ax=ax,
        space="SE2_GROUP",
        color="green",
        label="Right by Integration",
    )
    ax.set_aspect("equal")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()

# se2 = SpecialEuclidean(n=2, point_type="vector")

# X = gs.array([[0,0,0],
#                [0,0,1],
#                [0,1,0],
#                [1,0,0]])

# print(se2.metric.dist_pairwise(X))

# print((X[:,1:]**2).sum(axis=1))


# # metric = se2.left_canonical_metric

# def metric_matrix(point):
#     return gs.array([[0.1,0,0],
#               [0,gs.cos(point[...,0])**2 + 0.5*gs.sin(point[...,0])**2,(1-0.5)*gs.sin(point[...,0])*gs.cos(point[...,0])],
#               [0,(1-0.5)*gs.sin(point[...,0])*gs.cos(point[...,0]),gs.sin(point[...,0])**2 + 0.5*gs.cos(point[...,0])**2]])
    

# metric = SubRiemannianMetric(dim=3, dist_dim=3, frame = metric_matrix)

# print(metric.cometric_matrix(gs.array([3,0,0])))

# geo_points = metric.geodesic(initial_point=gs.array([0,0,0]), end_point=gs.array([1,0,0]))
# t = gs.linspace(0,1,40)
# print(geo_points(t))
