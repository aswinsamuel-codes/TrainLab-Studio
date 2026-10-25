"""Optional text-to-3D meshes for nicer-looking generated scenarios.

PLACEHOLDER: text-to-3D services (Meshy, Tripo, and similar) each have their own API.
Fill in ``GenericHTTPProvider.request_mesh`` for the service you sign up with, then set
in TrainLab/.env:

    TEXT_TO_3D_PROVIDER=generic
    TEXT_TO_3D_API_KEY=your-key-here
    TEXT_TO_3D_ENDPOINT=https://...

Meshes are visual only: physics keeps using the simple collision shapes, so training
speed and stability never depend on the downloaded model.
"""

from __future__ import annotations

import os
from pathlib import Path

from trainlab.generator.llm_client import PLACEHOLDER, load_env


class MeshProvider:
    name = "none"

    def available(self) -> bool:
        return False

    def make_mesh(self, description: str, out_dir: Path) -> Path | None:
        return None


class GenericHTTPProvider(MeshProvider):
    name = "generic"

    def __init__(self):
        self.key = os.environ.get("TEXT_TO_3D_API_KEY", "")
        self.endpoint = os.environ.get("TEXT_TO_3D_ENDPOINT", "")

    def available(self) -> bool:
        return bool(self.key and self.key != PLACEHOLDER and self.endpoint)

    def request_mesh(self, description: str) -> bytes:
        """Return GLB/OBJ bytes for ``description``. Adapt this to your provider's API."""
        raise NotImplementedError("Wire up your text-to-3D service here (see module docstring).")

    def make_mesh(self, description: str, out_dir: Path) -> Path | None:
        import io

        import trimesh

        data = self.request_mesh(description)
        mesh = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
        mesh.apply_scale(1.0 / max(mesh.extents))  # unit size; scale it in the MJCF
        out = out_dir / "visual.obj"
        mesh.export(out)
        return out


def get_mesh_provider() -> MeshProvider:
    load_env()
    if os.environ.get("TEXT_TO_3D_PROVIDER", "none").lower() == "generic":
        return GenericHTTPProvider()
    return MeshProvider()
