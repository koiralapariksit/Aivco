"""
AIVCO Server backward-compatibility shim.
Re-exports FastAPI application and server-level functions from server.app.
"""

from server.app import *
from server.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=True)
