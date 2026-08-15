"""Algorithm registry.

To add a new recommendation algorithm:
    1. Create a module in this package with a class implementing
       RecommendationAlgorithm (see base.py for the contract).
    2. Give it a unique `name` attribute.
    3. Import it below and add it to ALGORITHMS.

The experiment harness and CLI look algorithms up by name, so nothing else
needs to change.
"""

from .base import RecommendationAlgorithm, ProblemConfig
from .dreamteam import DreamTeamAlgorithm
from .random_baseline import RandomBaseline
from .combo import ComboAlgorithm
from .bocs import BocsAlgorithm

ALGORITHMS = {
    DreamTeamAlgorithm.name: DreamTeamAlgorithm,
    RandomBaseline.name: RandomBaseline,
    ComboAlgorithm.name: ComboAlgorithm,
    BocsAlgorithm.name: BocsAlgorithm,
    # Add new algorithms here, e.g.:
    # MyNewAlgorithm.name: MyNewAlgorithm,
}


def get_algorithm(name: str):
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {', '.join(sorted(ALGORITHMS))}"
        )
