"""
Sub-environment that builds a subset of ``{0, ..., n_options - 1}`` one element at a
time, in increasing order, and terminates with a STOP (EOS) action.

State: a list of ``n_options`` bits, ``state[c] == 1`` iff element ``c`` is in the
subset. The source is the empty subset.

Actions: ``(c + 1,)`` adds element ``c`` (1-based, as in Choice); ``(-1,)`` is STOP.

Canonical order: an element can only be added if it is larger than every element
already in the subset. Hence every subset is built by exactly one trajectory and
every non-source state has exactly one parent (remove its largest element), so the
backward policy is trivial. STOP is valid as soon as the subset is non-empty.

Only these structural rules are enforced here. Constraints that depend on the
context (e.g. which elements are available at a tree node, or which subsets are
valid splits) are added by the parent environment on top of this mask.
"""

from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat


class SubsetSequential(GFlowNetEnv):

    def __init__(self, n_options: int, **kwargs):
        assert n_options >= 1
        self.n_options = n_options
        self.source = [0] * n_options
        self.eos = (-1,)
        super().__init__(**kwargs)

    # Helpers
    @staticmethod
    def members(state: List[int]) -> List[int]:
        return [c for c, bit in enumerate(state) if bit] # return a list of the indices of the elements that are in the subset (i.e., where the bit is 1)

    def get_action_space(self) -> List[Tuple]:
        return [(c + 1,) for c in range(self.n_options)] + [self.eos] # action 1 is to add element 0, action 2 is to add element 1, ..., action n_options is to add element n_options-1, and action n_options+1 is to STOP

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
             Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [True] * self.action_space_dim # mask all action
        members = self.members(state)
        last = members[-1] if members else -1
        mask = [c <= last for c in range(self.n_options)] # create list by checking condition c <= last
        mask.append(len(members) == 0)  # len(members) == 0) is true when the subset is empty so it appends True to mask if the subset is empty so the eos action is masked
        # STOP needs a non-empty subset
        # mask for the empty state is [False, False, False, False, True]
        return mask






    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        # Returns (list of parent states, list of actions parent -> state).
        # Here there is always at most one parent (canonical increasing order).


        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [state], [self.eos]

        # Elements in the subset, e.g. [0, 1, 0, 1] -> [1, 3].
        members = self.members(state)

        # Empty subset = source state: nothing came before it, no parents.
        if not members:
            return [], []

        # Elements are added in increasing order, so the last one added is the
        # largest one. The only parent is the state without it.
        parent = list(state)  # copy, so the input state is not modified
        parent[members[-1]] = 0  # remove the largest element

        # Action that adds element c is (c + 1,).
        return [parent], [(members[-1] + 1,)]


    """
members = self.members([0, 1, 0, 1])   # [1, 3]
parent = list(state)                    # [0, 1, 0, 1]  (copy)
parent[members[-1]] = 0                 # parent[3] = 0 → [0, 1, 0, 0]
return [parent], [(members[-1] + 1,)]   # ([[0, 1, 0, 0]], [(4,)])

    """

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        do_step, self.state, action = self._pre_step(
            action=action,
            backward=False,
            skip_mask_check=skip_mask_check or self.skip_mask_check,
        )
        if not do_step:
            return self.state, action, False
        self.n_actions += 1

        if action == self.eos:
            self.done = True
            return self.state, action, True
        self.state[action[0] - 1] = 1
        
        return self.state, action, True




    def _get_max_trajectory_length(self) -> int:
        return self.n_options + 1

    def states2policy(self, states: List) -> TensorType["batch", "policy_input_dim"]:
        """The bit vector of each state."""
        return tfloat(states, float_type=self.float, device=self.device).reshape(
            len(states), self.n_options
        )

    def state2readable(self, state: Optional[List[int]] = None, alphabet={}) -> str:
        state = self._get_state(state)
        return "{" + ",".join(str(c) for c in self.members(state)) + "}"

    def readable2state(self, readable: str, alphabet={}) -> List[int]:
        state = [0] * self.n_options
        inner = readable.strip("{}")
        for c in inner.split(",") if inner else []:
            state[int(c)] = 1
        return state
