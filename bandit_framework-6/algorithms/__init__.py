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
from .neural_linear import NeuralLinearAlgorithm
from .neural_ucb_ts import NeuralUCBAlgorithm, NeuralTSAlgorithm
from .bocs_constrained import ConstrainedBocsAlgorithm
from .neural_linear_constrained import ConstrainedNeuralLinearAlgorithm
from .neural_ucb_constrained import ConstrainedNeuralUCBAlgorithm
from .neural_ts_constrained import ConstrainedNeuralTSAlgorithm
from .combo_constrained import ConstrainedComboAlgorithm

ALGORITHMS = {
    DreamTeamAlgorithm.name: DreamTeamAlgorithm,
    RandomBaseline.name: RandomBaseline,
    ComboAlgorithm.name: ComboAlgorithm,
    BocsAlgorithm.name: BocsAlgorithm,
    NeuralLinearAlgorithm.name: NeuralLinearAlgorithm,
    NeuralUCBAlgorithm.name: NeuralUCBAlgorithm,
    NeuralTSAlgorithm.name: NeuralTSAlgorithm,
    ConstrainedBocsAlgorithm.name: ConstrainedBocsAlgorithm,
    ConstrainedNeuralLinearAlgorithm.name: ConstrainedNeuralLinearAlgorithm,
    ConstrainedNeuralUCBAlgorithm.name: ConstrainedNeuralUCBAlgorithm,
    ConstrainedNeuralTSAlgorithm.name: ConstrainedNeuralTSAlgorithm,
    ConstrainedComboAlgorithm.name: ConstrainedComboAlgorithm,
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
