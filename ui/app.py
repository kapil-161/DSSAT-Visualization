"""
DSSAT Viewer Application
"""
import sys
import os
import signal
import logging
from dash import Dash
import dash_bootstrap_components as dbc
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

# Now import modules
from utils.dssat_paths import initialize_dssat_paths
import config
from ui.dash_thread import DashThread
from ui.layouts import create_app_layout
from ui.callbacks import register_callbacks


logger = logging.getLogger(__name__)

class DSSATViewer:
    """Main DSSAT visualization application.
    
    A Qt-based application that combines PyQt5 for the window management
    and Dash for the visualization interface.
    
    Attributes:
        qt_app (QApplication): The main Qt application instance
        view (QWebEngineView): The web view container for the Dash app
        dash_app (Dash): The Dash application instance
        dash_thread (DashThread): Thread running the Dash server
    """
    
    def __init__(self) -> None:
        try:
            # Validate configuration
            if not all([config.DASH_HOST, config.DASH_PORT]):
                raise ValueError("Invalid configuration: DASH_HOST and DASH_PORT must be set")
            
            # Initialize DSSAT paths
            initialize_dssat_paths()
            
            # Initialize PyQt application
            self.qt_app = QApplication(sys.argv)
            self.view = QWebEngineView()
            self.view.setWindowTitle("DSSAT Visualization")
            self.view.resize(1200, 800)
            
            # Initialize Dash app
            self.dash_app = Dash(
                __name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
            )
            
            self.setup_dash_app()
            
            # Setup server thread
            self.dash_thread = DashThread(self.dash_app)
            self.dash_thread.daemon = True
            self.dash_thread.start()
            
            # Setup window
            self.view.closeEvent = self.handle_close
            self.view.load(QUrl(f"http://{config.DASH_HOST}:{config.DASH_PORT}"))
            self.view.show()
            
        except Exception as e:
            logger.error(f"Error initializing DSSATViewer: {e}")
            raise

    def setup_dash_app(self) -> None:
        """Setup Dash application layout and callbacks.
        
        Raises:
            Exception: If there's an error setting up the layout or callbacks
        """
        try:
            self.dash_app.layout = create_app_layout()
            register_callbacks(self.dash_app)
            logger.info("Dash application setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up Dash app: {str(e)}", exc_info=True)
            raise

    def handle_close(self, event) -> None:
        """Handle close event for the Qt window.
        
        Args:
            event: The close event from Qt
        """
        try:
            logger.info("Shutting down application...")
            self.dash_thread.shutdown()
            event.accept()
            self.qt_app.quit()
            logger.info("Application shutdown completed")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
            sys.exit(1)

    def run(self):
        """Run the application."""
        try:
            # Setup signal handler for clean shutdown
            signal.signal(signal.SIGINT, lambda *args: self.qt_app.quit())
            
            # Start Qt application
            return self.qt_app.exec_()
        except Exception as e:
            logger.error(f"Error running application: {e}")
            return 1

def main():
    """Main entry point for the DSSAT Viewer application."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        app = DSSATViewer()
        return app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
