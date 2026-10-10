"""How strongly each layer is edited."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Kernel:
    """The strength peaks at max_weight on max_weight_position and falls linearly to min_weight
    min_weight_distance layers away. Layers further out are not edited.

    The field names match heretic's, so trials from its journals can be read directly.
    """

    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def at(self, layer: int) -> float:
        distance = abs(layer - self.max_weight_position)
        if distance > self.min_weight_distance:
            return 0.0
        return self.max_weight + distance / self.min_weight_distance * (self.min_weight - self.max_weight)
