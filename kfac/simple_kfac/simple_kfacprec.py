"""KFAC Preconditioner for GPT-NeoX."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Callable

import torch

from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.enums import ComputeMethod

from kfac.layers.register import register_modules
from kfac.simple_kfac.eigen4pipe import KFACPipeEigenLayer

logger = logging.getLogger(__name__)

class SimpleKFACPreconditioner(BaseKFACPreconditioner):
    """KFAC Distributed Gradient Preconditioner for GPT-NeoX.

    Integrates with DeepSpeed's PipelineModule that is used to enable
    3D parallelism in GPT-NeoX training.

    Implements the memory-optimized preconditioning scheme (gradient worker
    fraction of 1/world_size).
    """

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        *,
        factor_update_steps: Callable[[int], int] | int = 1,
        inv_update_steps: Callable[[int], int] | int = 1,
        # KFAC hyperparameters
        damping: Callable[[int], float] | float = 0.001,
        factor_decay: Callable[[int], float] | float = 0.95,
        kl_clip: Callable[[int], float] | float = 0.001,
        lr: Callable[[int], float] | float = 0.1,
        # Distribution strategy
        accumulation_steps: int = 1,
        compute_eigenvalue_outer_product: bool = False,
        symmetry_aware: bool = False,
        grad_scaler: (
            torch.cuda.amp.GradScaler | Callable[[], float] | None
        ) = None,
        factor_dtype: torch.dtype | None = None,
        inv_dtype: torch.dtype = torch.float32,
        factor_checkpoint_dir: str | None = None,
        skip_layers: list[str] | None = None,
        update_factors_in_hook: bool = True,
        loglevel: int = logging.DEBUG,
    ) -> None:
        """Init KFACPreconditioner.

        Args:
            model (torch.nn.Module): model to precondition with KFAC.
            factor_update_steps (Callable, int): steps between computing and
                updating the running average of the Kronecker factors or
                callable that takes the K-FAC step and returns the value.
            inv_update_steps (Callble, int): steps between recomputing and
                communicating the second-order information or callable that
                takes the K-FAC step and returns the value.
            damping (Callable, float): Tikhonov damping parameter or a callable
                that takes the K-FAC step and returns the damping parameter
                as a float (default: 0.001).
            factor_decay (Callable, float): running average coefficient for
                Kronecker factors or callable that takes the K-FAC step and
                returns the factor_decay (default: 0.95).
            kl_clip (Callable, float): clipping parameter for gradient scaling
                or a callable that takes the K-FAC step and returns a float.
                If None, no scaling/clipping will be applied (default: 0.001).
            lr (Callable, float): learning rate or callable that takes the
                K-FAC step and returns learning rate (default: 0.1).
            accumulation_steps (int): number of forward/backward passes
                between optimization steps (default: 1).
            allreduce_bucket_cap_mb (float): maximum size in megabytes for
                allreduce bucketing. If zero, bucketing is not used
                (default: 25).
            assignment_strategy (AssignmentStrategy, str): See
                `AssignmentStrategy` for more details
                (default: AssignmentStrategy.COMPUTE).
            compute_method (ComputeMethod, str): See `ComputeMethod` for more
                details (default: ComputeMethod.EIGEN).
            compute_eigenvalue_outer_product (bool): when using the eigen
                compute method, precompute the element-wise inverse of the
                outer product of eigenvectors on the eigen decomposition worker
                rather to reduce computation in the gradient preconditioning
                stage. `colocate_factors` must be True (default: True).
            symmetry_aware (bool): communicate only the upper triangle of
                symmetric matrices. Can reduce communication time when factors
                are large (default: False).
            data_parallel_group (ProcessGroup): DeepSpeed data parallel group.
            model_parallel_group (ProcessGroup): DeepSpeed model parallel
                group.
            pipeline_parallel_group (ProcessGroup): DeepSpeed pipeline parallel
                group.
            grad_scaler (torch.cuda.amp.GradScaler or callable): Gradient
                scaler used for Torch AMP training. Used to unscale the G
                factors as they are accumulated during the backward pass.
                Alternatively can be a callable which will return the current
                scale (default: None).
            factor_dtype (torch.dtype): force data type for storing factors.
                If None, defaults to data type of intermediate values in
                forward/backward pass (default: None).
            inv_dtype (torch.dtype): force data type for storing second-order
                data (e.g., inverses or eigen decompositions)
                (default: torch.float32).
            factor_checkpoint_dir (str): directory to store factors
                checkpoints in.
            skip_layers (list): list of module names to ignore when registering
                layers. Passing the name of parent modules will prevent
                recursively registering child modules of the parent.
                Case-insensitive (default: []).
            update_factors_in_hook (bool): If True, running average of factors
                is updated in the module hook and the async communication is
                started. Otherwise, this will be performed at the start of
                step() (default: True).
            loglevel (int): logging level (default: logging.DEBUG).
        """
        
        self.compute_eigenvalue_outer_product = (
            compute_eigenvalue_outer_product
        )
        self.compute_method = ComputeMethod.EIGEN
        self.grad_scaler = grad_scaler
        self.factor_dtype = factor_dtype
        self.inv_dtype = inv_dtype
        self.factor_checkpoint_dir = factor_checkpoint_dir
        self.skip_layers = [] if skip_layers is None else skip_layers
        self.symmetry_aware = symmetry_aware

        layer_kwargs = dict(
            grad_scaler=self.grad_scaler,
            factor_dtype=self.factor_dtype,
            inv_dtype=self.inv_dtype,
            symmetry_aware=self.symmetry_aware,
        )

        kfac_layers = register_modules(
            model,
            kfac_layer_type=KFACPipeEigenLayer,
            skip_layers=self.skip_layers,
            **layer_kwargs,
        )

        param2layer = {}
        for module, (name, layer) in kfac_layers.items():
            for name, param in module.named_parameters():
                param2layer[param] = (name, layer)

        defaults = {
            'compute_eigenvalue_outer_product': (
                self.compute_eigenvalue_outer_product
            ),
            'compute_method': self.compute_method,
            'grad_scaler': self.grad_scaler is not None,
            'factor_checkpoint_dir': self.factor_checkpoint_dir,
            'factor_dtype': self.factor_dtype,
            'inv_dtype': self.inv_dtype,
            'skip_layers': self.skip_layers,
            'symmetry_aware': self.symmetry_aware,
        }

        super().__init__(
            kfac_layers,
            factor_update_steps=factor_update_steps,
            inv_update_steps=inv_update_steps,
            factor_decay=factor_decay,
            damping=damping,
            kl_clip=kl_clip,
            lr=lr,
            accumulation_steps=accumulation_steps,
            assignment=None,
            update_factors_in_hook=update_factors_in_hook,
            defaults=defaults,
            tdc=None,
            loglevel=loglevel,
        )
        
    @torch.no_grad()
    def update_ag_inv(self) -> None:
        for name, layer in reversed(list(self._layers.values())):
            if self.steps % self.inv_update_steps == 0:
                layer.compute_a_inv(damping=self.damping)
                layer.compute_g_inv(damping=self.damping)
        self._steps += 1
        self._mini_steps = defaultdict(int)
        

    @torch.no_grad()
    def step(self) -> None:
        """Perform one K-FAC step.

        Note:
            This function should always be called before `optimizer.step()` as
            it modifies the gradients and does not modify the weights.

        Note:
            Gradients must be averaged across ranks before calling `step()`.
            This condition is guaranteed to be true if using the
            `DistributedDataParallel` model wrapper as gradients are
            communicated during `loss.backward()`.
        """
        # Compute Inverses
        for name, layer in reversed(list(self._layers.values())):
            if self.steps % self.inv_update_steps == 0:
                layer.compute_a_inv(damping=self.damping)
                layer.compute_g_inv(damping=self.damping)
            layer.preconditioned_grad(damping=self.damping)

        scale = None if self.kl_clip is None else self._compute_grad_scale()

        # Update gradients in-place
        for _, layer in reversed(list(self._layers.values())):
            layer.update_grad(scale=scale)

        self._steps += 1
        self._mini_steps = defaultdict(int)
