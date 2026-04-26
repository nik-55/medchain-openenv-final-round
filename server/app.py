"""
FastAPI application for the MedChain Env environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction

try:
    from .medchain_env_environment import MedchainEnvironment
    from ..models import MedchainToolObservation
except ImportError:
    from server.medchain_env_environment import MedchainEnvironment
    from models import MedchainToolObservation


def _env_factory():
    """Create a new MedchainEnvironment instance for each client session."""
    return MedchainEnvironment()


app = create_app(
    _env_factory,
    CallToolAction,
    MedchainToolObservation,
    env_name="medchain_env",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port) # Bypass multi node deployment requirement in openenv: main()
