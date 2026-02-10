"""
animation subpackage containing weight_animation and large weight animation.
"""
# For small weight animation
from .weight_animation import WeightMatrixAnime
from .weight_animation import animate_weight_heatmap

# For large weight animation, e.g. (1000 X 1000)
from .largewt_animation import LargeWeightMatrixAnime
from .largewt_animation import animate_large_heatmap