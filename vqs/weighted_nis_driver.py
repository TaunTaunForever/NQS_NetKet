"""NetKet driver for the target/proposal updates of weighted NIS."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from netket.driver import AbstractVariationalDriver
from netket.jax import tree_cast
from netket.optimizer import identity_preconditioner
from netket.utils import struct


class WeightedNISVMC(AbstractVariationalDriver):
    """A NetKet driver with explicit weighted NIS target and proposal updates.

    Target updates use ``WeightedNISState.expect_and_grad``.  Proposal updates
    fit the autoregressive proposal to the weighted pool produced by that same
    estimate.  NetKet's logger/callback/optimizer APIs remain available through
    ``AbstractVariationalDriver``.
    """

    # NetKet 3.22 makes optimisation drivers ``struct.Pytree`` instances.  All
    # attributes assigned by this driver must therefore be declared up-front.
    # Most are Python configuration or logging objects, so they are static
    # Pytree fields.  The proposal optimiser state and loss may contain JAX
    # arrays and remain dynamic fields.
    _ham: Any = struct.field(pytree_node=False, serialize=False)
    _preconditioner: Any = struct.field(pytree_node=False, serialize=False)
    proposal_optimizer: Any = struct.field(pytree_node=False, serialize=False)
    proposal_train_steps: int = struct.field(pytree_node=False, serialize=False)
    proposal_train_batch_size: int | None = struct.field(
        pytree_node=False, serialize=False
    )
    always_update_target: bool = struct.field(pytree_node=False, serialize=False)
    ess_threshold: float = struct.field(pytree_node=False, serialize=False)
    proposal_update_interval: int = struct.field(pytree_node=False, serialize=False)
    proposal_freeze_after: int | None = struct.field(
        pytree_node=False, serialize=False
    )
    heldout_diagnostics_every: int = struct.field(
        pytree_node=False, serialize=False
    )
    _proposal_optimizer_state: Any = struct.field(pytree_node=True, serialize=False)
    _proposal_loss: Any = struct.field(pytree_node=True, serialize=False)
    _proposal_updated: bool = struct.field(pytree_node=False, serialize=False)
    _target_updated: bool = struct.field(pytree_node=False, serialize=False)
    _heldout_diagnostics: dict[str, float | None] = struct.field(
        pytree_node=False, serialize=False
    )

    def __init__(
        self,
        hamiltonian,
        optimizer,
        *,
        variational_state,
        proposal_optimizer=None,
        proposal_train_steps: int = 1,
        proposal_train_batch_size: int | None = None,
        always_update_target: bool = True,
        ess_threshold: float = 0.0,
        preconditioner=None,
        proposal_update_interval: int = 1,
        proposal_freeze_after: int | None = None,
        heldout_diagnostics_every: int = 0,
    ):
        if variational_state.hilbert != hamiltonian.hilbert:
            raise TypeError("variational_state and hamiltonian must share a Hilbert space")
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")
        self._ham = hamiltonian.collect()
        self.preconditioner = identity_preconditioner if preconditioner is None else preconditioner
        self.proposal_optimizer = proposal_optimizer
        self.proposal_train_steps = int(proposal_train_steps)
        self.proposal_train_batch_size = proposal_train_batch_size
        self.always_update_target = bool(always_update_target)
        self.ess_threshold = float(ess_threshold)
        if self.ess_threshold < 0.0 or self.ess_threshold > 1.0:
            raise ValueError("ess_threshold must be between zero and one")
        self.proposal_update_interval = int(proposal_update_interval)
        if self.proposal_update_interval < 1:
            raise ValueError("proposal_update_interval must be positive")
        self.proposal_freeze_after = (
            None if proposal_freeze_after is None else int(proposal_freeze_after)
        )
        if self.proposal_freeze_after is not None and self.proposal_freeze_after < 0:
            raise ValueError("proposal_freeze_after must be non-negative or None")
        self.heldout_diagnostics_every = int(heldout_diagnostics_every)
        if self.heldout_diagnostics_every < 0:
            raise ValueError("heldout_diagnostics_every must be non-negative")
        self._proposal_optimizer_state = (
            None
            if proposal_optimizer is None
            else proposal_optimizer.init(self.state.proposal_parameters)
        )
        self._proposal_loss = None
        self._proposal_updated = False
        self._target_updated = False
        self._heldout_diagnostics: dict[str, float | None] = {}

    @property
    def preconditioner(self):
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        self._preconditioner = value

    @property
    def energy(self):
        return self._loss_stats

    @property
    def target_updated(self):
        """Whether the latest iteration applied a target-model update."""
        return self._target_updated

    @property
    def proposal_updated(self):
        """Whether the latest iteration fitted the proposal model."""
        return self._proposal_updated

    @property
    def heldout_diagnostics(self):
        """Independent-pool diagnostics from the latest scheduled check."""
        return dict(self._heldout_diagnostics)

    @staticmethod
    def _tree_norm(tree) -> float:
        return float(
            jax.device_get(
                jnp.sqrt(
                    sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree.leaves(tree))
                )
            )
        )

    def compute_loss_and_update(self):
        """Compute the weighted-NIS loss and target-model parameter update.

        NetKet 3.22 calls this after ``reset_step`` and expects the explicit
        ``(loss, update)`` pair.  The former ``_forward_and_backward`` hook
        only returned an update and is therefore intentionally not retained.
        """
        self._loss_stats, gradient = self.state.expect_and_grad(self._ham)
        ess_fraction = self.state.last_diagnostics["ESSFrac"]
        self._target_updated = self.always_update_target or bool(
            jax.device_get(ess_fraction) >= self.ess_threshold
        )
        current_step = int(self.step_count)
        self._proposal_updated = (
            self.proposal_optimizer is not None
            and self.proposal_train_steps > 0
            and current_step % self.proposal_update_interval == 0
            and (
                self.proposal_freeze_after is None
                or current_step < self.proposal_freeze_after
            )
        )
        self._proposal_loss = None
        if self._proposal_updated:
            _, self._proposal_optimizer_state, self._proposal_loss = self.state.train_proposal(
                self.proposal_optimizer,
                self._proposal_optimizer_state,
                n_steps=self.proposal_train_steps,
                batch_size=self.proposal_train_batch_size,
            )
        if not self._target_updated:
            gradient = jax.tree_util.tree_map(jnp.zeros_like, gradient)
        gradient = self.preconditioner(self.state, gradient, self.step_count)
        self._heldout_diagnostics = {}
        if self.heldout_diagnostics_every and (
            (current_step + 1) % self.heldout_diagnostics_every == 0
        ):
            heldout_stats, heldout_force, heldout_batch, heldout_nis = (
                self.state.evaluate_heldout_and_grad(self._ham)
            )
            heldout = {
                "Energy": float(jax.device_get(jnp.real(heldout_stats.mean))),
                "Variance": float(jax.device_get(heldout_stats.variance)),
                "ErrorOfMean": float(jax.device_get(heldout_stats.error_of_mean)),
                "ESS": float(jax.device_get(heldout_nis["ESS"])),
                "ESSFrac": float(jax.device_get(heldout_nis["ESSFrac"])),
                "ForceNorm": self._tree_norm(heldout_force),
            }
            heldout_residual = getattr(
                self.preconditioner, "heldout_relative_residual", None
            )
            if heldout_residual is not None:
                heldout["SRRelativeResidual"] = heldout_residual(
                    self.state, heldout_force, heldout_batch
                )
            self._heldout_diagnostics = heldout
        return self._loss_stats, tree_cast(gradient, self.state.parameters)

    def advance(self, steps: int = 1):
        """Advance manual weighted-NIS loops by one or more optimizer steps.

        The experiment runners perform a paired post-update energy check and
        possible rollback outside the NetKet driver.  NetKet 3.22 removed its
        old ``advance`` helper, so this compact equivalent preserves that
        workflow without relying on its deprecated driver hooks.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.reset_step()
            self._loss_stats, self._dp = self.compute_loss_and_update()
            self.update_parameters(self._dp)
            self._step_count += 1

    def _log_additional_data(self, log_dict):
        super()._log_additional_data(log_dict)
        diagnostics = self.state.last_diagnostics
        if diagnostics:
            log_dict["NIS"] = {
                name: float(jax.device_get(value)) for name, value in diagnostics.items()
            }
        if self._proposal_loss is not None:
            log_dict["ProposalLoss"] = float(jax.device_get(self._proposal_loss))
        sr_info = getattr(self.preconditioner, "last_info", None)
        if sr_info is not None:
            log_dict["WeightedSR"] = sr_info
        if self._heldout_diagnostics:
            log_dict["HeldOutNIS"] = self._heldout_diagnostics
