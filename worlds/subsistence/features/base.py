"""Feature lifecycle and dependency resolution."""

from __future__ import annotations

from typing import Any


class FeatureLayer:
    """One optional mechanism with a small lifecycle."""

    name: str = ""
    dependencies: tuple[str, ...] = ()

    def __init__(self, world: Any) -> None:
        self.world = world
        self.settings: dict[str, Any] = {}
        self.enabled = False

    def setup(self) -> None:
        """Read and validate this layer's configuration."""

    def before_regenerate(self) -> None:
        """Run before the commons regenerate."""

    def start_of_tick(self) -> None:
        """Run after regeneration and before gathering."""

    def after_gather(self) -> None:
        """Run after all living agents have worked the day."""

    def market(self) -> None:
        """Run the spot market, if the world has one."""

    def before_consume(self) -> None:
        """Run after exchange and before eating/spoilage."""

    def end_of_tick(self) -> None:
        """Run after consumption and spoilage."""

    def metrics(self, iteration: int) -> dict[str, Any]:
        return {}

    def default_metrics(self) -> dict[str, Any]:
        return {}


class PassiveLayer(FeatureLayer):
    """A flag whose behaviour is handled by the core world."""

    def __init__(
        self,
        world: Any,
        name: str,
        dependencies: tuple[str, ...] = (),
    ) -> None:
        super().__init__(world)
        self.name = name
        self.dependencies = dependencies


class FeatureRegistry:
    """Resolve flags with dependencies and dispatch lifecycle hooks."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.layers: dict[str, FeatureLayer] = {}
        self.effective: dict[str, bool] = {}

    def register(self, layer: FeatureLayer) -> None:
        if not layer.name:
            raise ValueError("A feature layer needs a name")
        if layer.name in self.layers:
            raise ValueError(f"Duplicate feature layer {layer.name!r}")
        self.layers[layer.name] = layer

    def resolve(self, requested: dict[str, bool]) -> dict[str, bool]:
        enabled = {
            name: bool(requested.get(name, False)) for name in self.layers
        }
        # Auto-enable declared dependencies. A small fixed-point loop keeps
        # this understandable even when layers are added later.
        changed = True
        while changed:
            changed = False
            for name, layer in self.layers.items():
                if not enabled.get(name):
                    continue
                for dependency in layer.dependencies:
                    if dependency in self.layers and not enabled[dependency]:
                        enabled[dependency] = True
                        changed = True
        self.effective = enabled
        for layer in self.layers.values():
            layer.enabled = enabled[layer.name]
        for layer in self.layers.values():
            if layer.enabled:
                layer.setup()
        return dict(enabled)

    def enabled(self, name: str) -> bool:
        return bool(self.effective.get(name, False))

    def layer(self, name: str) -> FeatureLayer:
        return self.layers[name]

    def run(self, hook: str) -> None:
        for layer in self.layers.values():
            if layer.enabled:
                getattr(layer, hook)()

    def metrics(self, iteration: int) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for layer in self.layers.values():
            merged.update(layer.default_metrics())
        for layer in self.layers.values():
            if layer.enabled:
                merged.update(layer.metrics(iteration))
        return merged
