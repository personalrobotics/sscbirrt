# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Node:
    """A node in the RRT tree."""

    config: np.ndarray
    parent: int | None = None  # Index of parent node, None for root
    source_index: Any = None  # Provenance of this root (input index or Sample.source path)


class RRTree:
    """Rapidly-exploring Random Tree data structure."""

    def __init__(
        self,
        root_config: np.ndarray | list[np.ndarray],
        source_indices: Sequence[Any] | None = None,
    ):
        """Initialize tree with one or more root nodes.

        Args:
            root_config: Single configuration or list of configurations.
                        All provided configs become roots (parent=None).
            source_indices: Optional index per root, tracking which input
                           (config or TSR) produced it. Same length as configs.
        """
        # Normalize to list
        if isinstance(root_config, np.ndarray):
            configs = [root_config]
        else:
            configs = root_config

        self.nodes: list[Node] = []
        self._configs: list[np.ndarray] = []
        self._configs_array: np.ndarray | None = None  # Cached for nearest()

        # Add all configs as roots
        for i, config in enumerate(configs):
            src_idx = source_indices[i] if source_indices is not None else i
            self.nodes.append(Node(config=config.copy(), parent=None, source_index=src_idx))
            self._configs.append(config.copy())

    def add_root(self, config: np.ndarray, source_index: Any = None) -> int:
        """Add a root: a child of the tree's virtual root, with no edge to check (CBiRRT's P_sample, #196).

        Every root hangs off one virtual root that is not a configuration, so the tree stays one tree however many
        start or goal members join it, before the search or during it.
        """
        self.nodes.append(Node(config=config.copy(), parent=None, source_index=source_index))
        self._configs.append(config.copy())
        self._configs_array = None  # Invalidate cache
        return len(self.nodes) - 1

    def add_node(self, config: np.ndarray, parent_idx: int) -> int:
        """Add a new node to the tree.

        Args:
            config: Joint configuration for new node
            parent_idx: Index of parent node

        Returns:
            Index of newly added node
        """
        node = Node(config=config.copy(), parent=parent_idx)
        self.nodes.append(node)
        self._configs.append(config.copy())
        self._configs_array = None  # Invalidate cache
        return len(self.nodes) - 1

    def nearest(self, config: np.ndarray) -> int:
        """Find the nearest node to a configuration.

        Args:
            config: Query configuration

        Returns:
            Index of nearest node
        """
        if self._configs_array is None or len(self._configs_array) != len(self._configs):
            self._configs_array = np.array(self._configs)
        distances = np.linalg.norm(self._configs_array - config, axis=1)
        return int(np.argmin(distances))

    def get_path_to_root(self, node_idx: int) -> list[np.ndarray]:
        """Extract path from node back to root.

        Args:
            node_idx: Index of node to trace from

        Returns:
            List of configurations from root to node
        """
        path = []
        idx = node_idx
        while idx is not None:
            path.append(self.nodes[idx].config)
            idx = self.nodes[idx].parent
        return list(reversed(path))

    def __len__(self) -> int:
        return len(self.nodes)

    def get_root_source_index(self, node_idx: int) -> Any:
        """Trace a node back to its root and return the source_index."""
        idx = node_idx
        while self.nodes[idx].parent is not None:
            idx = self.nodes[idx].parent
        return self.nodes[idx].source_index

    @property
    def num_roots(self) -> int:
        """Number of root nodes in the tree."""
        return sum(1 for node in self.nodes if node.parent is None)
