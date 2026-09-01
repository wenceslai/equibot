"""Task table for the robosuite / MimicGen viewpoint-shift experiment.

`gt_bodies` are the MuJoCo bodies whose geoms form the point cloud the policy
sees (GT segmentation — the oracle analogue of the SAM3 masks in the
object-centric pipeline). They mirror `TASK_SPECS` in the main repo's
`rerender_robomimic.py`, with one deliberate addition: `can` also gets the
TARGET bin (`bin2`). EquiBot centres the cloud and expresses the end-effector
relative to it, so absolute world position is unobservable — without the bin
in the cloud the policy cannot know where to carry the can. Square/stack
already contain their target (peg1 / the lower cube).
"""

import importlib

TASK_SPECS = {
    "can": {
        "env_name": "PickPlaceCan",
        "gt_bodies": ["Can_main", "bin2"],
        "url": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/demo_v141.hdf5",
        "n_demos_available": 200,
        "num_points": 256,   # paper App. E: Can uses 256-point clouds
        "train_demos": 200,  # all ph demos
    },
    "square": {
        "env_name": "NutAssemblySquare",
        "gt_bodies": ["SquareNut_main", "peg1"],
        "url": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/demo_v141.hdf5",
        "n_demos_available": 200,
        "num_points": 512,   # paper App. E: Square uses 512-point clouds
        "train_demos": 200,  # all ph demos
    },
    "square_d1": {
        "env_name": "Square_D1",
        "gt_bodies": ["SquareNut_main", "peg1"],
        "url": "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/square_d1.hdf5",
        "n_demos_available": 1000,
        "num_points": 512,   # = Square (paper App. E)
        "train_demos": 800,
    },
    "stack_d1": {
        "env_name": "Stack_D1",
        "gt_bodies": ["cubeA_main", "cubeB_main"],
        "url": "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d1.hdf5",
        "n_demos_available": 1000,
        "num_points": 1024,  # not in the paper; their default for all other tasks
        "train_demos": 800,
    },
    "stack_three_d1": {
        "env_name": "StackThree_D1",
        "gt_bodies": ["cubeA_main", "cubeB_main", "cubeC_main"],
        "url": "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_three_d1.hdf5",
        "n_demos_available": 1000,
        "num_points": 1024,  # not in the paper; their default for all other tasks
        "train_demos": 800,
    },
}

# Envs defined by MimicGen rather than robosuite. `import mimicgen` registers
# them; the value is the env module to force-import when the registry lookup
# misses, so the underlying exception surfaces instead of mimicgen's
# swallowed warning.
MIMICGEN_ENVS = {
    "Square_D1": "mimicgen.envs.robosuite.nut_assembly",
    "Stack_D1": "mimicgen.envs.robosuite.stack",
    "StackThree_D1": "mimicgen.envs.robosuite.stack",
}

_MIMICGEN_INSTALL_HINT = (
    "    git clone https://github.com/NVlabs/mimicgen && "
    "pip install -e mimicgen --no-deps\n"
    "(--no-deps: mimicgen does not support robosuite v1.5+ and a plain install "
    "would move the robosuite pin.)"
)


def register_mimicgen_envs(env_name: str):
    """No-op for plain robosuite envs; imports mimicgen for MimicGen ones and
    verifies the env actually reached robosuite's registry."""
    if env_name not in MIMICGEN_ENVS:
        return
    from robosuite.environments.base import REGISTERED_ENVS
    try:
        import mimicgen  # noqa: F401  — the import IS the registration
    except ImportError as exc:
        raise SystemExit(
            f"env '{env_name}' is a MimicGen env, but `import mimicgen` "
            f"failed: {exc}\nInstall it into THIS env with:\n"
            f"{_MIMICGEN_INSTALL_HINT}"
        )
    if env_name in REGISTERED_ENVS:
        return
    if getattr(mimicgen, "__file__", None) is None:
        raise SystemExit(
            "`import mimicgen` resolved to a namespace package with no code "
            "— almost always a `git clone` of mimicgen in the current "
            "directory shadowing the installed package. Run from elsewhere "
            f"or reinstall:\n{_MIMICGEN_INSTALL_HINT}"
        )
    module = MIMICGEN_ENVS[env_name]
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(
            f"mimicgen is installed at {mimicgen.__file__} but importing "
            f"{module} failed: {type(exc).__name__}: {exc}\n"
            f"{_MIMICGEN_INSTALL_HINT}"
        )
    if env_name not in REGISTERED_ENVS:
        raise SystemExit(
            f"'{env_name}' still not registered after importing {module}. "
            f"Registered: {sorted(REGISTERED_ENVS)}"
        )


if __name__ == "__main__":
    # `python -m equibot.envs.robosuite_sim.tasks url can` -> prints the URL
    # (used by the experiment script for downloads).
    import sys
    what, task = sys.argv[1], sys.argv[2]
    print(TASK_SPECS[task][what])
