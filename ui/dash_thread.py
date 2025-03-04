"""
Thread implementation for running Dash server.

This module provides a threaded server implementation for Dash applications,
allowing the Dash server to run concurrently with other application components.
"""

import sys
import os
import threading
from typing import Any
from werkzeug.serving import make_server
from dash import Dash

# Add project root to Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

import config

class DashThread(threading.Thread):
    """Thread for running Dash server.
    
    Attributes:
        server: Werkzeug server instance
        ctx: Flask application context
        
    Args:
        app (Dash): The Dash application instance to serve
    """
    
    def __init__(self, app: Dash) -> None:
        super().__init__()
        self.server = make_server(
            host=config.DASH_HOST,
            port=config.DASH_PORT,
            app=app.server
        )
        self.ctx = app.server.app_context()
        self.ctx.push()
        self._is_running = True

    def run(self) -> None:
        """Start the server and run forever until shutdown is called."""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)
            self._is_running = False
            raise

    def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self._is_running:
            self.server.shutdown()
            self.ctx.pop()
            self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if the server is running.
        
        Returns:
            bool: True if server is running, False otherwise
        """
        return self._is_running
