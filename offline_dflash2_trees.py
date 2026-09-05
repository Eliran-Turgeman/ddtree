import heapq
from dataclasses import dataclass
from typing import Protocol

import torch


UNARY_FULL_MASS = "Unary-FullMass"
UNARY_TRUNCATED = "Unary-Truncated"
PAIRWISE_MASS_PRESERVING = "Pairwise-MassPreserving"
PAIRWISE_AFTER_ROOT = "Pairwise-after-root"


@dataclass(frozen=True)
class TreeNode:
    token_id: int
    candidate_index: int
    depth: int
    parent: int
    log_prefix_score: float
    path_candidate_indices: tuple[int, ...]


class PrefixScorer(Protocol):
    name: str
    depth: int
    candidate_count: int

    def extension_log_score(
        self,
        depth_index: int,
        predecessor_candidate_index: int | None,
        candidate_index: int,
    ) -> float:
        ...

    def maximum_extension_log_score(self) -> float:
        ...


class UnaryScorer:
    def __init__(
        self,
        name: str,
        log_probabilities: torch.Tensor,
    ) -> None:
        self.name = name
        probabilities = log_probabilities.float().cpu()
        self.depth, self.candidate_count = probabilities.shape
        self._maximum_extension = float(probabilities.max())
        self.log_probabilities = probabilities.tolist()

    def extension_log_score(
        self,
        depth_index: int,
        predecessor_candidate_index: int | None,
        candidate_index: int,
    ) -> float:
        del predecessor_candidate_index
        return self.log_probabilities[depth_index][candidate_index]

    def maximum_extension_log_score(self) -> float:
        return self._maximum_extension


class PairwiseMassPreservingScorer:
    name = PAIRWISE_MASS_PRESERVING

    def __init__(
        self,
        log_retained_mass: torch.Tensor,
        anchor_final_scores: torch.Tensor,
        pairwise_final_scores: torch.Tensor,
    ) -> None:
        retained_mass = log_retained_mass.float().cpu()
        anchor_log_conditional = torch.log_softmax(
            anchor_final_scores.float(),
            dim=-1,
        ).cpu()
        pairwise_log_conditional = torch.log_softmax(
            pairwise_final_scores.float(),
            dim=-1,
        ).cpu()
        self.depth = retained_mass.shape[0]
        self.candidate_count = anchor_log_conditional.shape[0]
        anchor_extensions = retained_mass[0] + anchor_log_conditional
        pairwise_extensions = (
            retained_mass[1:, None, None] + pairwise_log_conditional
        )
        self.anchor_extensions = anchor_extensions.tolist()
        self.pairwise_extensions = pairwise_extensions.tolist()
        maximum = anchor_extensions.max()
        self._pairwise_maximum = float("-inf")
        if pairwise_extensions.numel():
            pairwise_maximum = pairwise_extensions.max()
            self._pairwise_maximum = float(pairwise_maximum)
            maximum = torch.maximum(maximum, pairwise_maximum)
        self._maximum_extension = float(maximum)

    def extension_log_score(
        self,
        depth_index: int,
        predecessor_candidate_index: int | None,
        candidate_index: int,
    ) -> float:
        if depth_index == 0:
            return self.anchor_extensions[candidate_index]
        if predecessor_candidate_index is None:
            raise ValueError(
                "pairwise scoring requires a predecessor after depth 1"
            )
        return self.pairwise_extensions[depth_index - 1][
            predecessor_candidate_index
        ][candidate_index]

    def maximum_extension_log_score(self) -> float:
        return self._maximum_extension


class PairwiseAfterRootScorer(PairwiseMassPreservingScorer):
    name = PAIRWISE_AFTER_ROOT

    def __init__(
        self,
        root_log_probabilities: torch.Tensor,
        log_retained_mass: torch.Tensor,
        anchor_final_scores: torch.Tensor,
        pairwise_final_scores: torch.Tensor,
    ) -> None:
        super().__init__(
            log_retained_mass,
            anchor_final_scores,
            pairwise_final_scores,
        )
        root_probabilities = root_log_probabilities.float().cpu()
        self.root_log_probabilities = root_probabilities.tolist()
        self._maximum_extension = max(
            float(root_probabilities.max()),
            self._pairwise_maximum,
        )

    def extension_log_score(
        self,
        depth_index: int,
        predecessor_candidate_index: int | None,
        candidate_index: int,
    ) -> float:
        if depth_index == 0:
            return self.root_log_probabilities[candidate_index]
        return super().extension_log_score(
            depth_index,
            predecessor_candidate_index,
            candidate_index,
        )

    def maximum_extension_log_score(self) -> float:
        return self._maximum_extension


def validate_lattice_tensors(
    lattice: dict,
    *,
    expected_depth: int | None = None,
    expected_candidate_count: int | None = None,
    probability_tolerance: float = 1e-5,
) -> tuple[int, int]:
    depth, candidate_count = validate_unary_lattice_tensors(
        lattice,
        expected_depth=expected_depth,
        expected_candidate_count=expected_candidate_count,
        probability_tolerance=probability_tolerance,
    )
    anchor_final_scores = lattice["anchor_final_scores"]
    pairwise_final_scores = lattice["pairwise_final_scores"]

    expected_shapes = {
        "anchor_final_scores": (candidate_count,),
        "pairwise_final_scores": (
            max(depth - 1, 0),
            candidate_count,
            candidate_count,
        ),
    }
    tensors = {
        "anchor_final_scores": anchor_final_scores,
        "pairwise_final_scores": pairwise_final_scores,
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(tensors[name].shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(tensors[name].shape)}"
            )
    return depth, candidate_count


def validate_unary_lattice_tensors(
    lattice: dict,
    *,
    expected_depth: int | None = None,
    expected_candidate_count: int | None = None,
    probability_tolerance: float = 1e-5,
) -> tuple[int, int]:
    candidate_token_ids = lattice["candidate_token_ids"]
    unary_logits = lattice["candidate_unary_logits"]
    unary_logsumexp = lattice["unary_logsumexp"]

    if candidate_token_ids.ndim != 2:
        raise ValueError(
            "candidate_token_ids must have shape [depth, candidates], "
            f"got {tuple(candidate_token_ids.shape)}"
        )
    depth, candidate_count = candidate_token_ids.shape
    expected_shapes = {
        "candidate_unary_logits": (depth, candidate_count),
        "unary_logsumexp": (depth,),
    }
    tensors = {
        "candidate_unary_logits": unary_logits,
        "unary_logsumexp": unary_logsumexp,
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(tensors[name].shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(tensors[name].shape)}"
            )
    if expected_depth is not None and depth != expected_depth:
        raise ValueError(
            f"candidate depth must be {expected_depth}, got {depth}"
        )
    if (
        expected_candidate_count is not None
        and candidate_count != expected_candidate_count
    ):
        raise ValueError(
            "candidate count must be "
            f"{expected_candidate_count}, got {candidate_count}"
        )
    log_retained_mass = (
        torch.logsumexp(unary_logits.float(), dim=-1)
        - unary_logsumexp.float()
    )
    if bool(torch.any(log_retained_mass > probability_tolerance)):
        raise ValueError(
            "retained candidate probability mass exceeds one: "
            f"maximum log mass {float(log_retained_mass.max())}"
        )
    return depth, candidate_count


def build_scorer(
    trace_round: dict,
    name: str,
    *,
    validate: bool = True,
) -> PrefixScorer:
    if validate:
        if name in (UNARY_FULL_MASS, UNARY_TRUNCATED):
            validate_unary_lattice_tensors(trace_round)
        else:
            validate_lattice_tensors(trace_round)
    unary_logits = trace_round["candidate_unary_logits"].float()
    unary_logsumexp = trace_round["unary_logsumexp"].float()
    retained_logsumexp = torch.logsumexp(unary_logits, dim=-1)
    log_retained_mass = retained_logsumexp - unary_logsumexp

    if name == UNARY_FULL_MASS:
        return UnaryScorer(
            UNARY_FULL_MASS,
            unary_logits - unary_logsumexp.unsqueeze(-1),
        )
    if name == UNARY_TRUNCATED:
        return UnaryScorer(
            UNARY_TRUNCATED,
            torch.log_softmax(unary_logits, dim=-1),
        )
    if name == PAIRWISE_MASS_PRESERVING:
        return PairwiseMassPreservingScorer(
            log_retained_mass,
            trace_round["anchor_final_scores"],
            trace_round["pairwise_final_scores"],
        )
    if name == PAIRWISE_AFTER_ROOT:
        return PairwiseAfterRootScorer(
            unary_logits[0] - unary_logsumexp[0],
            log_retained_mass,
            trace_round["anchor_final_scores"],
            trace_round["pairwise_final_scores"],
        )
    raise ValueError(f"unknown scorer {name!r}")


def build_scorers(trace_round: dict) -> dict[str, PrefixScorer]:
    validate_lattice_tensors(trace_round)
    return {
        name: build_scorer(trace_round, name, validate=False)
        for name in (
            UNARY_FULL_MASS,
            UNARY_TRUNCATED,
            PAIRWISE_MASS_PRESERVING,
            PAIRWISE_AFTER_ROOT,
        )
    }


def build_best_first_tree(
    candidate_token_ids: torch.Tensor,
    scorer: PrefixScorer,
    budget: int,
    *,
    monotonic_tolerance: float = 1e-5,
) -> list[TreeNode]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    candidate_token_ids = candidate_token_ids.long().cpu()
    depth, candidate_count = candidate_token_ids.shape
    if depth != scorer.depth or candidate_count != scorer.candidate_count:
        raise ValueError(
            "candidate lattice shape does not match scorer: "
            f"{tuple(candidate_token_ids.shape)} vs "
            f"({scorer.depth}, {scorer.candidate_count})"
        )
    if budget == 0 or depth == 0:
        return []

    frontier: list[
        tuple[
            float,
            tuple[int, ...],
            int,
            int,
            int,
            float,
        ]
    ] = []
    for candidate_index in range(candidate_count):
        log_score = scorer.extension_log_score(
            0,
            None,
            candidate_index,
        )
        if log_score > monotonic_tolerance:
            raise ValueError(
                "non-monotonic root extension: "
                f"log probability {log_score}"
            )
        log_score = min(log_score, 0.0)
        path = (candidate_index,)
        heapq.heappush(
            frontier,
            (
                -log_score,
                path,
                -1,
                1,
                candidate_index,
                log_score,
            ),
        )

    nodes: list[TreeNode] = []
    while frontier and len(nodes) < budget:
        (
            _,
            path,
            parent,
            node_depth,
            candidate_index,
            log_prefix_score,
        ) = heapq.heappop(frontier)
        node_index = len(nodes)
        token_id = int(
            candidate_token_ids[node_depth - 1, candidate_index]
        )
        nodes.append(
            TreeNode(
                token_id=token_id,
                candidate_index=candidate_index,
                depth=node_depth,
                parent=parent,
                log_prefix_score=log_prefix_score,
                path_candidate_indices=path,
            )
        )

        if node_depth >= depth:
            continue
        next_depth_index = node_depth
        for child_candidate_index in range(candidate_count):
            extension_score = scorer.extension_log_score(
                next_depth_index,
                candidate_index,
                child_candidate_index,
            )
            child_log_score = log_prefix_score + extension_score
            if child_log_score > log_prefix_score + monotonic_tolerance:
                raise ValueError(
                    "non-monotonic prefix score: "
                    f"parent={log_prefix_score}, child={child_log_score}"
                )
            child_log_score = min(child_log_score, log_prefix_score)
            child_path = path + (child_candidate_index,)
            heapq.heappush(
                frontier,
                (
                    -child_log_score,
                    child_path,
                    node_index,
                    node_depth + 1,
                    child_candidate_index,
                    child_log_score,
                ),
            )

    return nodes


def target_candidate_path(
    candidate_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[list[int], int]:
    candidate_token_ids = candidate_token_ids.long().cpu()
    target_token_ids = target_token_ids.long().cpu()
    path = []
    max_depth = min(
        candidate_token_ids.shape[0],
        target_token_ids.numel(),
    )
    for depth_index in range(max_depth):
        matches = torch.nonzero(
            candidate_token_ids[depth_index]
            == target_token_ids[depth_index],
            as_tuple=True,
        )[0]
        if matches.numel() == 0:
            break
        path.append(int(matches[0]))
    return path, len(path)


def prefix_entry_budgets(
    nodes: list[TreeNode],
    target_path: list[int],
) -> list[int | None]:
    entry_by_path = {
        node.path_candidate_indices: rank
        for rank, node in enumerate(nodes, start=1)
    }
    return [
        entry_by_path.get(tuple(target_path[:depth]))
        for depth in range(1, len(target_path) + 1)
    ]


def matched_draft_tokens(
    entry_budgets: list[int | None],
    budget: int,
) -> int:
    matched = 0
    for entry_budget in entry_budgets:
        if entry_budget is None or entry_budget > budget:
            break
        matched += 1
    return matched


def greedy_path_matched_tokens(
    selected_token_ids: list[int] | torch.Tensor,
    target_token_ids: list[int] | torch.Tensor,
    budget: int,
) -> int:
    matched = 0
    for selected_token, target_token in zip(
        selected_token_ids,
        target_token_ids,
    ):
        if int(selected_token) != int(target_token):
            break
        matched += 1
        if matched == budget:
            break
    return matched
