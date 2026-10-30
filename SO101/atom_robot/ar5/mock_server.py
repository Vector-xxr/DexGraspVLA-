"""A stand-in VLA server for bring-up.

Speaks the same ``POST /act`` contract as ``vla-scripts/deploy.py`` but needs no
checkpoint and no GPU, so the client, the cameras, the arm and the hand can all
be validated before the policy exists.

    python atom_robot/ar5/mock_server.py --port 8777

Then, in another shell:

    python atom_robot/ar5/run_ar5_eval.py --mock True --num_rollouts_planned 1

The returned actions are a slow circular sweep with a periodic grasp, not
anything task-directed -- the point is to exercise the transport, the action
encoding and the safety clamps.
"""

import argparse
import logging
from typing import Any, Dict

import json_numpy
import numpy as np

json_numpy.patch()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ACTION_DIM = 7
CHUNK = 8


class MockVLAServer:
    def __init__(self, amplitude: float = 0.006, period: int = 60):
        self.amplitude = amplitude
        self.period = period
        self.calls = 0

    # The Dict annotation is required: without it FastAPI treats `payload` as a
    # query parameter and the request body never reaches the handler.
    def act(self, payload: Dict[str, Any]):
        self.calls += 1

        # Validate the payload exactly as the real server's get_vla_action does,
        # so a malformed client is caught here rather than at model-load time.
        missing = [k for k in ("full_image", "state", "instruction") if k not in payload]
        if missing:
            logger.error(f"payload missing {missing}; got keys {sorted(payload)}")
            # deploy.py signals failure with a bare "error" string; match it so
            # the client's error handling is exercised identically.
            return "error"

        wrist_keys = [k for k in payload if "wrist" in k]
        if self.calls == 1:
            logger.info(
                f"instruction={payload['instruction']!r} "
                f"full_image={np.asarray(payload['full_image']).shape} "
                f"wrist={wrist_keys} state={np.asarray(payload['state']).shape}"
            )

        t0 = (self.calls - 1) * CHUNK
        actions = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
        for i in range(CHUNK):
            phase = 2 * np.pi * ((t0 + i) % self.period) / self.period
            actions[i, 0] = self.amplitude * np.cos(phase)
            actions[i, 1] = self.amplitude * np.sin(phase)
            actions[i, 2] = 0.002 * np.sin(phase / 2)
            actions[i, 5] = 0.01 * np.cos(phase)
            # Closed for the middle third of each cycle.
            actions[i, 6] = 0.0 if self.period / 3 <= ((t0 + i) % self.period) < 2 * self.period / 3 else 1.0

        # Hand back the raw array, exactly as deploy.py does: json_numpy.patch()
        # teaches the default JSON encoder about ndarrays. Wrapping it in
        # json_numpy.dumps() here would double-encode and the client would
        # receive a string.
        return JSONResponse(actions)

    def run(self, host: str, port: int) -> None:
        app = FastAPI()
        app.post("/act")(self.act)
        logger.info(f"Mock VLA server on http://{host}:{port}/act -- chunk={CHUNK}, action_dim={ACTION_DIM}")
        uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--amplitude", type=float, default=0.006, help="Per-step translation, meters")
    args = parser.parse_args()
    MockVLAServer(amplitude=args.amplitude).run(args.host, args.port)


if __name__ == "__main__":
    main()
