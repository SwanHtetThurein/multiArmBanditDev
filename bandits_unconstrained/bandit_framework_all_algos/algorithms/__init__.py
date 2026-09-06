"""Algorithm registry.

To add a new recommendation algorithm:
    1. Create a module in this package with a class implementing
       RecommendationAlgorithm (see base.py for the contract).
    2. Give it a unique `name` attribute.
    3. Import it below and add it to ALGORITHMS.

The experiment harness and CLI look algorithms up by name, so nothing else
needs to change.

The registry is grouped by family, which is also the grouping used for
comparisons:

  reference points
    random        uniform random every round; performance floor
    dreamteam       this project's modified algorithm (per-bandit Thompson
                    sampling + type-scheduled stickiness + global switch budget,
                    with the 0.1 threshold / +1000 beta penalty update)
    dreamteam_orig  DreamTeam exactly as published (Zhou, Valentine & Bernstein,
                    CHI 2018): standard Beta update and equal-share global
                    constraint. Run against `dreamteam` to measure what this
                    project's modifications changed.

  model-free combinatorial search  (baselines in BOCS / COMBO)
    sa            simulated annealing with a geometric cooling schedule
    ols           oblivious local search: sweep the neighbourhood, move, restart
    regevo        regularized (aging) evolution over a population of teams

  surrogate-model Bayesian optimization
    bocs          sparse Bayesian linear model w/ pairwise terms + Thompson
    bocs_hs       ...the same, with the paper's real horseshoe prior (Gibbs)
    combo         GP with a diffusion kernel on the graph Cartesian product + EI
    combo_slice   ...the same, with the paper's real slice-sampled hyperparameters
    smac          random-forest surrogate + EI (tree surrogate, no kernel)
    gp_onehot     textbook GP-EI on a one-hot encoding (the "naive GP" ablation)
    cocabo        per-dimension EXP3 agents + GP with a categorical overlap kernel

  neural surrogates
    neurallinear  Bayesian linear posterior on an MLP's last hidden layer
    bootnn        bootstrap ensemble of MLPs, act greedily w.r.t. one member

  acquisition rules on ONE shared surrogate
    (all four use the identical second-order Bayesian linear model as `bocs`,
     so differences between them isolate the acquisition rule alone)
    bocs          Thompson sampling            (listed above)
    linucb        upper confidence bound
    kg            knowledge gradient (one-step-optimal, decision-theoretic)
    purexp        no acquisition at all -- uniform allocation, model-based
                  recommendation only

  fixed-budget best-arm identification
    bayesgap      gap-based allocation aimed at the final recommendation
                  rather than at reward earned along the way

  combinatorial bandits  (base arm = one (bandit, arm) pair, exact oracle)
    cucb          optimistic per-base-arm estimates
    cts           Thompson sampling per base arm
"""

from .base import RecommendationAlgorithm, ProblemConfig

# reference points
from .dreamteam import DreamTeamAlgorithm
from .dreamteam_original import DreamTeamOriginal
from .random_baseline import RandomBaseline

# model-free combinatorial search
from .simulated_annealing import SimulatedAnnealing
from .local_search import ObliviousLocalSearch
from .regularized_evolution import RegularizedEvolution

# surrogate-model Bayesian optimization
from .combo import ComboAlgorithm
from .combo_slice import ComboSlice
from .bocs import BocsAlgorithm
from .bocs_horseshoe import BocsHorseshoe
from .smac import SmacAlgorithm
from .gp_onehot import GPOneHotAlgorithm
from .cocabo import CoCaBO

# neural surrogates
from .neural_linear import NeuralLinearAlgorithm
from .bootstrapped_nn import BootstrappedNN

# acquisition rules on the shared second-order linear surrogate
from .lin_ucb import LinUCBAlgorithm
from .knowledge_gradient import KnowledgeGradient
from .pure_exploration import PureExploration

# fixed-budget best-arm identification
from .bayes_gap import BayesGap

# combinatorial bandits
from .cucb import CUCBAlgorithm
from .combinatorial_ts import CombinatorialTS

ALGORITHMS = {
    DreamTeamAlgorithm.name: DreamTeamAlgorithm,
    DreamTeamOriginal.name: DreamTeamOriginal,
    RandomBaseline.name: RandomBaseline,

    SimulatedAnnealing.name: SimulatedAnnealing,
    ObliviousLocalSearch.name: ObliviousLocalSearch,
    RegularizedEvolution.name: RegularizedEvolution,

    ComboAlgorithm.name: ComboAlgorithm,
    ComboSlice.name: ComboSlice,
    BocsAlgorithm.name: BocsAlgorithm,
    BocsHorseshoe.name: BocsHorseshoe,
    SmacAlgorithm.name: SmacAlgorithm,
    GPOneHotAlgorithm.name: GPOneHotAlgorithm,
    CoCaBO.name: CoCaBO,

    NeuralLinearAlgorithm.name: NeuralLinearAlgorithm,
    BootstrappedNN.name: BootstrappedNN,

    LinUCBAlgorithm.name: LinUCBAlgorithm,
    KnowledgeGradient.name: KnowledgeGradient,
    PureExploration.name: PureExploration,

    BayesGap.name: BayesGap,

    CUCBAlgorithm.name: CUCBAlgorithm,
    CombinatorialTS.name: CombinatorialTS,
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
