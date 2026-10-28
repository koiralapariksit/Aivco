"""
AIVCO Server Package
FastAPI application, REST endpoints, and WebSocket bridges.
"""

from server.app import *
from server.app import app

__all__ = ["app"]
