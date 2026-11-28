"""
Node of a decision tree that splits on a categorical feature with a subset of
categories built SEQUENTIALLY.

A DecisionTreeNodeCategoricalSequential represents a single decision node
consisting of two stages:
1. Choosing a feature (via a Choice sub-environment).
2. Building the left set ``L`` of categories of that feature, one category per
   step in increasing order, then STOP (via a SubsetSequential sub-environment).

The resulting split is ``x[feature] in L``: samples whose category is in ``L`` go
to the left child, all others to the right child.

Categories are integer codes ``0, ..., k_f - 1`` for feature ``f``, where ``k_f`` is
``n_categories[f]``. ``L`` is stored as a bit list (one bit per category) and, as a
split value, as a bitmask: category ``c`` is in ``L`` iff bit ``c`` is 1.

Which categories can be chosen depends on the ancestors of the node in the tree:
an ancestor ``x[f] in L`` keeps only ``A & L`` in its left subtree and ``A - L`` in
its right subtree. The parent Tree computes these "available categories" ``A`` and
publishes them on the node via :py:meth:`set_available_categories`, the categorical
counterpart of the threshold bounds of the numeric nodes. The masks then enforce:

- a feature can only be chosen if it has at least 2 available categories (otherwise
  the split would send every sample to the same side);
- the first category added is ``min(A)`` (forced: ``L`` and ``A - L`` are the same
  split with the children swapped, so the first available category always goes
  left);
- only categories of ``A`` can be added, in increasing order (one trajectory per
  ``L``);
- a category cannot be added if ``L`` would become ``A`` (empty right child).

This gives ``2^(|A| - 1) - 1`` splits per feature, each reached by exactly one
trajectory, with ``k + 1`` policy outputs for the subset stage (``k`` categories +
STOP), at the cost of ``|L| + 1`` steps per split.

This mirrors :py:class:`DecisionTreeNodeDiscreteChoice`, and keeps the attribute and
method names the Tree relies on (``get_threshold``, ``threshold_min``,
``set_threshold_bounds``, ...). For this node the "threshold" is the bitmask of L.
"""

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import numpy.typing as npt

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.envs.tree.subset_sequential import SubsetSequential
from gflownet.utils.common import copy

class DecisionTreeNodeCategoricalSequential(Stack):

    # Read by the Tree (and proxy) to route samples with the node's goes_left
    # instead of ``<=``
    is_categorical = True

    MAX_CATEGORIES = 64

    def __init__(
        self,
        features: Sequence,
        n_categories: Sequence[int],
        **kwargs,
    ):
        """
        Parameters
        ----------
        features : Sequence
            The set of features.
        n_categories : Sequence of int
            Number of categories of each feature, in the same order as
            ``features``. Feature ``f`` takes the integer codes
            ``0, ..., n_categories[f] - 1``.
        """
        n_categories = [int(k) for k in n_categories]
        assert len(n_categories) == len(features), (
            f"n_categories has {len(n_categories)} entries but there are "
            f"{len(features)} features."
        )  # making sure if we have n features then for each features we have an associated categories number
        assert all(k >= 1 for k in n_categories), "Every feature needs a category."
        assert max(n_categories) <= self.MAX_CATEGORIES, (
            f"At most {self.MAX_CATEGORIES} categories per feature are supported."
        )

        self.features = features
        self.n_features = len(features)
        self.n_categories = n_categories
        self.max_categories = max(n_categories) # so if there is a feature with 3 categories and another with 5 categories, max_categories will be 5
        # and the subset stage has 5 slots, but for the 2nd feature only 3 can be added, the rest will be masked

        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_category = 1

        # Alias used by the Tree and by the tests written for the numeric nodes
        self.stage_threshold = self.stage_category
        # Stage 1: build L one category at a time, then STOP
        self.category = SubsetSequential(n_options=self.max_categories, **kwargs)
        # Available categories per feature (0-based codes), set by the parent Tree
        # before computing each mask. Initially every category is available.
        self._available: List[Set[int]] = self._all_categories()
        super().__init__(subenvs=tuple([self.feature, self.category]), **kwargs)

    def _all_categories(self) -> List[Set[int]]:
        return [set(range(k)) for k in self.n_categories]

    # Sub-environments
    @property # function decorator to make a method behave like an attribute, so that we can access it like an attribute without parentheses
    def feature_env(self) -> Choice:
        """Returns the Choice sub-environment used for feature selection."""
        return self.subenvs[self.stage_feature]

    @property
    def category_env(self) -> SubsetSequential:
        """Returns the SubsetSequential sub-environment that builds L."""
        return self.subenvs[self.stage_category]

    # Range of the "threshold" (here: bitmask of L), used by
    # Tree._get_threshold_bounds. Categorical nodes ignore the bounds.
    @property
    def threshold_min(self) -> float:
        return 0.0

    @property
    def threshold_max(self) -> float:
        return float(2**self.max_categories - 1)

    # Bitmask <-> set conversion
    #
    # A subset of categories is stored as ONE integer (a bitmask): category c
    # is in the subset iff bit c is 1. Each category has a value 2^c:
    #   red (0) = 1, blue (1) = 2, green (2) = 4, yellow (3) = 8
    # and a subset is the sum of its values, e.g. {red, green} = 1 + 4 = 5.
    # Every subset gives a different number, so the number IS the subset.

    @staticmethod
    def mask_to_set(mask: int) -> Set[int]:
        # Number -> set: keep every category c whose bit is 1.
        # (mask >> c) & 1 reads bit c, e.g. 5 = 0b101 -> bits 0 and 2 -> {0, 2}
        return {c for c in range(mask.bit_length()) if (mask >> c) & 1}

    @staticmethod
    def set_to_mask(categories) -> int:
        # Set -> number: add 2^c for each category (1 << c == 2^c),
        # e.g. {0, 2} -> 1 + 4 = 5
        return sum(1 << int(c) for c in categories)

    #read-only functions of a state: they take a state, read part of it,

    # State helpers , during the training we need to know which feature and category is selected, so we need to extract them from the state
    def get_feature(self, state: Optional[Dict] = None) -> Optional[int]:
        """
        Returns the selected feature index (1-based) from the state, or None if no
        feature has been selected yet.
        """

        # _get_state (gflownet/envs/base.py): if a state is passed, use it;
        # if state is None, use the env's current state (self.state).
        # Here the state is the whole node, e.g.
        #   {'_active': 1, 0: [2], 1: [1, 0, 0, 0]}
        state = self._get_state(state) # from base.py to get the state of the env here the node

        # _get_substate (gflownet/envs/composite/base.py, inherited via Stack):
        # extracts the state of one sub-env from the node state.
        # stage_feature = 0, so this returns state[0], the feature Choice state,
        # e.g. [2].
        feature_state = self._get_substate(state, self.stage_feature)

        # A Choice's source state is [0] = nothing selected yet.
        # In that case no feature has been chosen, so return None.
        if self.feature_env.is_source(feature_state):
            return None  # if it returns none it mean that the node is at the very beginning

        # Otherwise the Choice state is [i], where i is the chosen option (1-based):
        # 1 = color, 2 = size, 3 = shape. Return i (still 1-based; use
        # self.features[i - 1] to get the name).
        return feature_state[0]

    '''
    node state = {'_active': 1, 0: [2], 1: [1, 0, 0, 0]}
    _get_substate(state, 0) → [2]
    is_source([2])?          → no  ([0] would be the source)
    return 2                 → feature = size

    if  feature state is [0] (source) → return None then
    valid action are over the features (1,2,3) and L is not started yet
    feature state = [0]   →  valid next actions: (0, 0, 1), (0, 0, 2), (0, 0, 3)
                                              color      size       shape
    '''

    def get_category(self, state: Optional[Dict] = None) -> Optional[int]:
        """
        Returns the bitmask of the (possibly partial) left subset ``L``, or None
        if ``L`` is still empty.
        """
        # Example: {'_active': 1, 0: [1], 1: [1, 0, 1, 0]}
        #   stage 1 active, feature 1 = color, bits [1, 0, 1, 0] -> L = {0, 2}
        #   -> bitmask 1 + 4 = 5
        state = self._get_state(state)

        # '_active' = which stage of the Stack is being built (0 = feature,
        # 1 = subset). If we are still in the feature stage (active < 1), L has
        # not started.
        if state.get("_active", 0) < self.stage_category: # stage_category is 1
            return None # still at feature stage

        # Extract the SubsetSequential sub-state: state[1], one bit per category
        bits = self._get_substate(state, self.stage_category)
        mask = sum(1 << c for c, bit in enumerate(bits) if bit)
        # All bits 0 = no category added yet
        return mask if mask > 0 else None

    def get_threshold(self, state: Optional[Dict] = None) -> Optional[int]:
        """
        Returns the split value of the node, which for this node is the bitmask
        of L. Kept under this name so that the Tree, the proxy and the
        evaluation code can extract splits from any node type.
        """
        # The Tree, the proxy and eval_tree call node_env.get_threshold(...) on
        # every node type (tree.py: _threshold_bucket, constraints, display;
        # proxy/tree.py:54; eval_tree.py:135). They don't know whether the node
        # is numeric or categorical.
        # For numeric nodes this returns the threshold t (split: x <= t).
        # For this node it returns the bitmask of L (split: x in L), so the
        # value is a bitmask, not a threshold.
        return self.get_category(state)

    # Split interface used by the Tree, the proxy and the evaluation code.

    def get_split_categories(self, state: Optional[Dict] = None) -> Optional[Set[int]]:
        """
        Returns the set of 0-based codes sent to the LEFT child (L), or None if
        L is empty. Used by the Tree to compute the available categories of the
        children: left child A & L, right child A - L.
        """
        mask = self.get_category(state)
        return None if mask is None else self.mask_to_set(mask)

    def goes_left(self, values: npt.NDArray, split_value) -> npt.NDArray:
        """Routing rule given the split value (bitmask) from get_threshold."""
        # A sample with category x goes left iff bit x of the mask is 1.
        # Example with mask 5 = {red, green}:
        #   x = 0 (red)   -> (5 >> 0) & 1 = 1 -> left
        #   x = 1 (blue)  -> (5 >> 1) & 1 = 0 -> right
        #   x = 2 (green) -> (5 >> 2) & 1 = 1 -> left
        # Done on the whole numpy array at once (no Python loop).
        values = np.asarray(values).astype(np.int64)
        return ((int(split_value) >> values) & 1).astype(bool)

    def format_split(self, split_value) -> str:
        """Short text of the split for plots, e.g. ``"in {0,2}"``."""
        return "in {" + ",".join(map(str, sorted(self.mask_to_set(int(split_value))))) + "}"

    # State conversion
    # Readable format: "Feature: color; Subset: 0,2"
    def state2readable(self, state: Optional[Dict] = None) -> str:
        """
        Converts a state into a human-readable string.

        Format: ``"Feature: <name>; Subset: <codes>"``. If either has not been
        selected yet, ``"<pending>"`` is shown.
        """
        state = self._get_state(state)
        feature_idx = self.get_feature(state)
        mask = self.get_category(state)
        feature_name = (
            str(self.features[feature_idx - 1]) if feature_idx is not None else "<pending>"
        )
        subset = (
            ",".join(map(str, sorted(self.mask_to_set(mask))))
            if mask is not None
            else "<pending>"
        )
        return f"Feature: {feature_name}; Subset: {subset}"

    def readable2state(self, readable: str) -> Dict:
        """
        Converts a human-readable representation of a state back into environment
        format.
        """
        parts = readable.split("; ")
        feature_part = parts[0].split(": ", 1)[1]
        subset_part = parts[1].split(": ", 1)[1]

        if feature_part == "<pending>":
            feature_state = copy(self.feature_env.source)
        else:
            feature_state = [list(self.features).index(feature_part) + 1]

        bits = copy(self.category_env.source)
        if subset_part != "<pending>":
            for c in subset_part.split(","):
                bits[int(c)] = 1

        if feature_part == "<pending>":
            active = 0
        elif subset_part == "<pending>":
            active = 1
        else:
            active = self.n_subenvs

        state = {"_active": active}
        state[self.stage_feature] = feature_state
        state[self.stage_category] = bits
        return state

    # Category-constraint 

    def set_available_categories(self, available: List[Set[int]]) -> None:
        """Stores the available categories per feature for the next masks."""
        self._available = [set(a) for a in available]

    def clear_available_categories(self) -> None:
        """Makes every category of every feature available again."""
        self._available = self._all_categories()

    def valid_choice_indices(
        self, feature_idx: int, state: Optional[Dict] = None
    ) -> Set[int]:
        """
        1-based "add category" actions of the SubsetSequential stage that are
        valid (STOP excluded), given the node ``state``.
        """
        # A = categories of this feature that can still reach this node
        available = sorted(self._available[feature_idx - 1])
        if len(available) < 2:
            return set()
        bits = self._get_substate(self._get_state(state), self.stage_category)
        members = {c for c, bit in enumerate(bits) if bit}
        if not members:
            return {available[0] + 1}  # forced first category: min(A) goes left
        last = max(members)
        # Only categories of A larger than the last one added (increasing order),
        # and never all of A (the right child must not be empty)
        return {
            c + 1
            for c in available
            if c > last and members | {c} != set(available)
        }

    def can_split(self, available: List[Set[int]]) -> bool:
        """True if at least one feature has 2 or more available categories."""
        return any(len(a) >= 2 for a in available)

    def set_threshold_bounds(self, lower: float, upper: float) -> None:
        pass

    def clear_threshold_bounds(self) -> None:
        self.clear_available_categories()

    def apply_threshold_rescale(self, substate: Dict, lower: float, upper: float) -> Dict:
        return substate

    def unapply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        return substate





    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward mask of the Stack and additionally invalidates:

        - in the feature stage, features with fewer than 2 available categories;
        - in the subset stage, "add category" actions that are not valid for the
          chosen feature (see :py:meth:`valid_choice_indices`).
        """
        state = self._get_state(state)
        mask = list(super().get_mask_invalid_actions_forward(state, done))

        active_subenv = self._get_active_subenv(state)
        if active_subenv == self.stage_feature:
            choice = self.feature_env
            invalid = {
                f
                for f in range(1, self.n_features + 1)
                if len(self._available[f - 1]) < 2
            }
        elif active_subenv == self.stage_category:
            choice = self.category_env
            feature_idx = self.get_feature(state)
            valid = self.valid_choice_indices(feature_idx, state)
            invalid = set(range(1, self.max_categories + 1)) - valid
        else:
            return mask


        #   [one-hot subenv idx (n_subenvs)] + [active subenv mask] + [padding].
        offset = self.n_subenvs
        for i, action in enumerate(choice.action_space):
            if action != choice.eos and action[0] in invalid:
                mask[offset + i] = True
        return mask




    def get_valid_actions(
        self,
        mask: Optional[List[bool]] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ):
        """
        Returns the list of valid actions, ensuring that the customized forward mask
        is used when no mask is supplied.
        """
        if mask is None and not backward:
            mask = self.get_mask_invalid_actions_forward(state, done)
        return super().get_valid_actions(
            mask=mask, state=state, done=done, backward=backward
        )



