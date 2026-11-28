"""
Tests for the Tree environment with categorical nodes
(``node_type="categorical_sequential"``).

- ``Tree(node_type="categorical_sequential", node_kwargs={"features": [...],
  "n_categories": [k_1, ..., k_d]})``. When the tree is built from data,
  ``n_categories`` is derived from the training set.

- A node is ``Stack(Choice(feature), SubsetSequential(max(n_categories)))``.
  The feature Choice is 1-based (0 means "not set"). The subset stage builds the
  left set ``L`` one category per step: action ``(c + 1,)`` adds code ``c``,
  ``(-1,)`` is STOP.

- ``node_env.get_category(node_state)`` returns the bitmask of ``L``
  (category ``c`` in ``L`` iff bit ``c`` is 1).

- Split rule: ``x[feature] in L`` goes to the LEFT child, everything else to the
  right.

- Masks, with ``A`` the categories of the feature available at the node:
    * the first category added is ``min(A)`` (mirror rule: ``L`` and ``A - L``
      are the same split with the children swapped);
    * only categories of ``A`` can be added, in increasing order;
    * ``L`` can never become ``A`` (empty right child);
    * categories that do not exist for the chosen feature are never in ``A``;
    * an ancestor ``x[f] in L`` keeps ``A & L`` on the left and ``A - L`` on the
      right;
    * features with fewer than 2 available categories cannot be chosen; for ex
      after ``color in {blue}``, every sample on the left is blue, so color has 1
      available category there and splitting on it again would send everything
      left and nothing right, a useless split. So color is masked in that subtree.

"""

from copy import deepcopy
from itertools import combinations

import common
import numpy as np
import pandas as pd
import pytest

from gflownet.envs.tree.tree import Tree
from gflownet.proxy.tree import _route_samples_to_leaves

NODE_TYPE = "categorical_sequential"
FEATURES = ["color", "size", "shape"]
N_CATEGORIES = [4, 3, 2]  # color: red/blue/green/yellow, size: 3, shape: 2

# 1-based feature Choice indices
COLOR, SIZE, SHAPE = 1, 2, 3
# 0-based data codes
RED, BLUE, GREEN, YELLOW = 0, 1, 2, 3


def mask(*codes):
    """Bitmask of a set of categories, e.g. mask(RED, GREEN) = 1 + 4 = 5."""
    return sum(1 << c for c in codes)


def _make_tree(max_depth, n_categories=None, **kwargs):
    return Tree(
        max_depth=max_depth,
        node_type=NODE_TYPE,
        node_kwargs={
            "features": FEATURES,
            "n_categories": n_categories or N_CATEGORIES,
        },
        **kwargs,
    )


@pytest.fixture
def env_cat_depth1():
    return _make_tree(1)


@pytest.fixture
def env_cat_depth2():
    return _make_tree(2)


@pytest.fixture
def env_cat_depth3():
    return _make_tree(3)


@pytest.fixture
def categorical_csv(tmp_path):
    """Small all-categorical dataset in the CSV format of ``Tree._load_dataset``."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "color": rng.integers(0, 4, n),
            "size": rng.integers(0, 3, n),
            "shape": rng.integers(0, 3, n),
        }
    )
    # Make sure every category appears in the training set
    df.loc[:3, "color"] = [0, 1, 2, 3]
    df.loc[:2, "size"] = [0, 1, 2]
    df.loc[:2, "shape"] = [0, 1, 2]
    df["class"] = (df["color"] == 1).astype(int)
    df["Split"] = ["train"] * 48 + ["test"] * 12
    path = tmp_path / "cat.csv"
    df.to_csv(path, index=False)
    return path


parametrize_cat_envs = pytest.mark.parametrize(
    "envs", ["env_cat_depth1", "env_cat_depth2", "env_cat_depth3"]
)
parametrize_cat_envs_bigger_than_depth_1 = pytest.mark.parametrize(
    "envs", ["env_cat_depth2", "env_cat_depth3"]
)


# Helpers


def _start_node(tree, node_idx, feature_idx):
    """Activates a node and chooses its feature, leaving L empty."""
    for action in [tree._pad_action((node_idx,), -1), (0, 0, feature_idx), (0, 0, -1)]:
        _, _, v = tree.step(action)
        assert v, f"Invalid action {action} starting node {node_idx}"


def _add(tree, code):
    """Adds the (0-based) category ``code`` to L. Returns whether it was valid."""
    return tree.step((0, 1, code + 1))[2]


def _build_node(tree, node_idx, feature_idx, codes):
    """
    Builds a complete node: activate, choose feature, feature EOS, add the
    categories of L in increasing order, STOP, deactivate.
    """
    _start_node(tree, node_idx, feature_idx)
    for c in sorted(codes):
        assert _add(tree, c), f"Failed to add category {c} in node {node_idx}"
    for action in [(0, 1, -1), tree._pad_action((node_idx,), -1)]:
        _, _, v = tree.step(action)
        assert v, f"Invalid action {action} finishing node {node_idx}"


def _valid_adds(tree):
    """0-based codes that can be added to L, and whether STOP is valid."""
    actions = [a for a in tree.get_valid_actions() if a[0] == 0 and a[1] == 1]
    adds = {a[2] - 1 for a in actions if a[2] != -1}
    return adds, (0, 1, -1) in actions


def _valid_features(tree):
    """1-based feature indices that are valid actions in the current state."""
    return {
        a[2]
        for a in tree.get_valid_actions()
        if a[0] == 0 and a[1] == 0 and a[2] != -1
    }


# Counting trees (target of the uniform-reward log Z test)


def _splits(available):
    """Valid left sets L for one feature with available categories ``available``."""
    a = sorted(available)
    if len(a) < 2:
        return []
    first, rest = a[0], a[1:]
    return [
        {first, *combo}
        for r in range(len(rest))  # excludes taking all of ``rest``
        for combo in combinations(rest, r)
    ]


def count_trees(n_categories, max_depth):
    """
    Independent pure-Python count (no env code) of the distinct trees with a
    split at the root:
    T(avail, d) = 1 + sum over splits of T(left avail, d-1) * T(right avail, d-1),
    T(., 0) = 1, and the answer is T(root, D) - 1 (the empty tree is excluded).
    """

    def T(avail, d):
        if d == 0:
            return 1
        total = 1
        for f, a in enumerate(avail):
            for left in _splits(a):
                avail_l = [set(x) for x in avail]
                avail_r = [set(x) for x in avail]
                avail_l[f] = a & left
                avail_r[f] = a - left
                total += T(avail_l, d - 1) * T(avail_r, d - 1)
        return total

    return T([set(range(k)) for k in n_categories], max_depth) - 1


def count_env_trees(env):
    """
    Brute-force count of the distinct terminal trees by depth-first search over
    all valid forward actions.

    From depth 2 on, the same tree is reached by several trajectories (the
    children of a node can be built in either order), so terminal states are
    deduplicated by their readable representation.
    """
    env.reset()
    trees = set()

    def dfs(state, done):
        if done:
            trees.add(env.state2readable(state))
            return
        env.set_state(deepcopy(state), done=False)
        for action in env.get_valid_actions():
            env.set_state(deepcopy(state), done=False)
            next_state, _, valid = env.step(action)
            assert valid
            dfs(deepcopy(next_state), env.done)

    dfs(deepcopy(env.source), False)
    return len(trees)


def test__count_env_trees__matches_formula_for_discrete_choice():
    """
    Sanity check of the brute-force counter on Tim's discrete_choice tree,
    where at depth 1 every split is a tree: d = n_features * n_thresholds.
    """
    env = Tree(
        max_depth=1,
        node_type="discrete_choice",
        node_kwargs={"features": ["a", "b"], "n_thresholds": 3},
    )
    assert count_env_trees(env) == 2 * 3


def test__count_trees__formula_hand_checked():
    # depth 1: 2^(k-1) - 1 splits per feature: (2^3-1) + (2^2-1) + (2^1-1)
    assert count_trees(N_CATEGORIES, 1) == 7 + 3 + 1


@pytest.mark.parametrize("max_depth", [1, 2])
def test__env_tree_count_matches_formula(max_depth):
    """11 trees at depth 1 (log Z -> log 11), 659 at depth 2 (log Z -> log 659)."""
    env = _make_tree(max_depth)
    assert count_env_trees(env) == count_trees(N_CATEGORIES, max_depth)


# Initialization


@parametrize_cat_envs
def test__categorical__environment_initializes(envs, request):
    env = request.getfixturevalue(envs)
    assert env.node_type == NODE_TYPE
    assert env.is_categorical
    assert env.continuous is False
    assert env.is_source()


def test__categorical__subset_stage_has_one_slot_per_category(env_cat_depth1):
    assert env_cat_depth1.node_env.n_categories == N_CATEGORIES
    assert env_cat_depth1.node_env.category_env.n_options == max(N_CATEGORIES)


def test__categorical__n_categories_must_match_features():
    with pytest.raises((AssertionError, ValueError)):
        _make_tree(1, n_categories=[4, 3])


def test__categorical__too_many_categories_raises():
    with pytest.raises(AssertionError):
        _make_tree(1, n_categories=[65, 3, 2])


@parametrize_cat_envs
def test__categorical__build_root_and_terminate(envs, request):
    env = request.getfixturevalue(envs)
    _build_node(env, 0, COLOR, {RED, GREEN})
    assert env._node_is_done(0, env.state)
    assert env.node_env.get_feature(env.state[0]) == COLOR
    assert env.node_env.get_category(env.state[0]) == mask(RED, GREEN)
    _, _, valid = env.step(env.eos)
    assert valid
    assert env.done


@parametrize_cat_envs
def test__categorical__forward_then_backward_reaches_source(envs, request):
    env = request.getfixturevalue(envs)
    _build_node(env, 0, COLOR, {RED, GREEN})
    env.step(env.eos)
    while not env.is_source():
        a = env.eos if env.done else env.get_valid_actions(backward=True)[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


# Masks: building L at the root


def test__categorical__first_category_is_forced(env_cat_depth1):
    """L is empty: only min(A) = red can be added, and STOP is not valid yet."""
    env = env_cat_depth1
    _start_node(env, 0, COLOR)
    assert _valid_adds(env) == ({RED}, False)


def test__categorical__canonical_order_and_stop(env_cat_depth1):
    env = env_cat_depth1
    _start_node(env, 0, COLOR)
    assert _add(env, RED)
    assert _valid_adds(env) == ({BLUE, GREEN, YELLOW}, True)
    assert _add(env, GREEN)
    # blue is now below the last added category: not allowed any more
    assert _valid_adds(env) == ({YELLOW}, True)
    assert _add(env, BLUE) is False


def test__categorical__cannot_fill_all_available(env_cat_depth1):
    """{red, blue, green} + yellow would leave the right child empty."""
    env = env_cat_depth1
    _start_node(env, 0, COLOR)
    for c in (RED, BLUE, GREEN):
        assert _add(env, c)
    assert _valid_adds(env) == (set(), True)


def test__categorical__nonexistent_categories_are_invalid(env_cat_depth1):
    """``size`` has 3 categories, so code 3 can never be added."""
    env = env_cat_depth1
    _start_node(env, 0, SIZE)
    assert _add(env, 0)
    assert _valid_adds(env) == ({1, 2}, True)
    assert _add(env, 3) is False


def test__categorical__two_categories_single_split(env_cat_depth1):
    """``shape`` has 2 categories: the only split is {0} vs {1}."""
    env = env_cat_depth1
    _start_node(env, 0, SHAPE)
    assert _valid_adds(env) == ({0}, False)
    assert _add(env, 0)
    assert _valid_adds(env) == (set(), True)


# Masks: ancestor constraints


@parametrize_cat_envs_bigger_than_depth_1
def test__categorical__children_available_categories(envs, request):
    """Root color in {red, green}: left keeps {red, green}, right {blue, yellow}."""
    env = request.getfixturevalue(envs)
    _build_node(env, 0, COLOR, {RED, GREEN})
    assert env._get_available_categories(1, env.state)[0] == {RED, GREEN}
    assert env._get_available_categories(2, env.state)[0] == {BLUE, YELLOW}
    _start_node(env, 2, COLOR)  # right child of root
    # Two categories left: only {blue} (the first available) is a valid split
    assert _valid_adds(env) == ({BLUE}, False)
    assert _add(env, BLUE)
    assert _valid_adds(env) == (set(), True)


@parametrize_cat_envs_bigger_than_depth_1
def test__categorical__exhausted_feature_is_masked(envs, request):
    """Root ``shape in {0}``: shape (k=2) has a single category in each child."""
    env = request.getfixturevalue(envs)
    _build_node(env, 0, SHAPE, {0})
    env.step(env._pad_action((1,), -1))  # activate left child of root
    assert SHAPE not in _valid_features(env)
    assert {COLOR, SIZE} <= _valid_features(env)


@parametrize_cat_envs_bigger_than_depth_1
def test__categorical__other_feature_not_constrained(envs, request):
    """Constraints only apply to ancestors splitting on the same feature."""
    env = request.getfixturevalue(envs)
    _build_node(env, 0, COLOR, {RED, GREEN})
    _start_node(env, 1, SIZE)  # left child, different feature
    assert env._get_available_categories(1, env.state)[1] == {0, 1, 2}
    assert _valid_adds(env) == ({0}, False)
    assert _add(env, 0)
    assert _valid_adds(env) == ({1, 2}, True)


def test__categorical__constraints_accumulate_down_the_tree(env_cat_depth3):
    """
    Root ``color in {red}`` (right: blue/green/yellow), then its right child
    ``color in {blue}`` (right: green/yellow). Only 2 categories remain in
    node 6, so the only valid split is {green}.
    """
    env = env_cat_depth3
    _build_node(env, 0, COLOR, {RED})
    _build_node(env, 2, COLOR, {BLUE})
    assert env._get_available_categories(6, env.state)[0] == {GREEN, YELLOW}
    _start_node(env, 6, COLOR)  # right child of node 2
    assert _valid_adds(env) == ({GREEN}, False)


# Routing

X_ROUTING = np.array([[RED, 0, 0], [BLUE, 1, 1], [GREEN, 2, 0], [YELLOW, 0, 1]])


def test__categorical__env_routing(env_cat_depth1):
    env = env_cat_depth1
    _build_node(env, 0, COLOR, {RED, GREEN})
    leaves = env._route_samples(env.state, X_ROUTING)
    assert sorted(leaves[Tree.left_child_idx(0)]) == [0, 2]
    assert sorted(leaves[Tree.right_child_idx(0)]) == [1, 3]


def test__categorical__proxy_routing(env_cat_depth1):
    """The proxy has its own (vectorized) copy of the routing logic."""
    env = env_cat_depth1
    _build_node(env, 0, COLOR, {RED, GREEN})
    leaves = _route_samples_to_leaves(env, env.state, X_ROUTING)
    assert sorted(leaves[Tree.left_child_idx(0)]) == [0, 2]
    assert sorted(leaves[Tree.right_child_idx(0)]) == [1, 3]


def test__categorical__env_and_proxy_routing_agree_depth2(env_cat_depth2):
    env = env_cat_depth2
    _build_node(env, 0, COLOR, {RED, GREEN})
    _build_node(env, 2, SIZE, {0, 2})
    rng = np.random.default_rng(0)
    X = np.stack([rng.integers(0, k, 200) for k in N_CATEGORIES], axis=1)
    env_leaves = env._route_samples(env.state, X)
    proxy_leaves = _route_samples_to_leaves(env, env.state, X)
    assert {k: sorted(v) for k, v in env_leaves.items()} == {
        k: sorted(v) for k, v in proxy_leaves.items()
    }
    # Spot-check the semantics
    for i, x in enumerate(X):
        if x[0] in (RED, GREEN):
            expected = Tree.left_child_idx(0)
        elif x[1] in (0, 2):
            expected = Tree.left_child_idx(2)
        else:
            expected = Tree.right_child_idx(2)
        assert i in env_leaves[expected]


# Readable format


def test__categorical__readable_roundtrip(env_cat_depth2):
    env = env_cat_depth2
    _build_node(env, 0, COLOR, {RED, GREEN, YELLOW})
    assert "Subset: 0,2,3" in env.state2readable()
    # Tree.readable2state is not implemented (Tim's TODO), so test the node
    # (``_active`` of a done node is not recovered, as in Tim's discrete_choice
    # node: it compares the feature and subset sub-states)
    node, substate = env.node_env, env.state[0]
    parsed = node.readable2state(node.state2readable(substate))
    assert parsed[0] == substate[0] and parsed[1] == substate[1]


# Data loading


def test__categorical__n_categories_derived_from_data(categorical_csv):
    env = Tree(max_depth=2, node_type=NODE_TYPE, data_path=str(categorical_csv))
    assert env.node_env.n_categories == [4, 3, 3]


def test__categorical__columns_are_not_scaled(categorical_csv):
    """Integer codes must survive loading unchanged (no min-max scaling)."""
    env = Tree(max_depth=2, node_type=NODE_TYPE, data_path=str(categorical_csv))
    df = pd.read_csv(categorical_csv)
    X_expected = df[df["Split"] == "train"][["color", "size", "shape"]].values
    assert np.array_equal(env.X_train, X_expected)


# Generic test suite


class TestTreeCategoricalDepth1(common.BaseTestsContinuous):
    """Common tests for the categorical Tree with depth 1."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cat_depth1):
        self.env = env_cat_depth1
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestTreeCategoricalDepth2(common.BaseTestsContinuous):
    """Common tests for the categorical Tree with depth 2."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cat_depth2):
        self.env = env_cat_depth2
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
