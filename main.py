"""
DSSAT Viewer - Main entry point
Updated with optimizations for faster startup in executable form
"""
import sys
import os
import warnings
import traceback
import logging
from pathlib import Path
from utils.lazy_loader import LazyLoader
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QMessageBox
# Lazy load heavy modules
pd = LazyLoader('pandas')
np = LazyLoader('numpy')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add constants for window dimensions and positioning
WINDOW_CONFIG = {
    'width': 1200,
    'height': 800,
    'min_width': 800,
    'min_height': 600
}

# Configure startup optimizations first
try:
    from optimized_startup import optimize_qt_settings
    optimize_qt_settings()
except ImportError:
    pass

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add the project root directory to the Python path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

def center_window(window):
    screen = QApplication.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )

def create_application():
    """Create and configure the QApplication instance."""
    if not QApplication.instance():
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Use Fusion style for better performance
        
        # Defer stylesheet loading
        app.setProperty("defer_stylesheet", True)
        return app
    return QApplication.instance()

from splash_screen import show_splash

if __name__ == "__main__":
    try:
        # Set Qt attributes first
        from optimized_startup import optimize_qt_settings
        optimize_qt_settings()
        
        # Create application instance and show splash screen
        app, splash = show_splash()
        app.processEvents()  # Ensure splash is displayed
        
        # Import and initialize main application
        try:
            from ui.app import DSSATViewer
            viewer = DSSATViewer()
            
            # Configure window
            viewer.view.resize(WINDOW_CONFIG['width'], WINDOW_CONFIG['height'])
            viewer.view.setMinimumSize(
                QSize(WINDOW_CONFIG['min_width'], WINDOW_CONFIG['min_height'])
            )
            
            # Center window
            center_window(viewer.view)
            
            # Show main window and close splash
            viewer.view.show()
            splash.finish(viewer.view)
            sys.exit(app.exec_())
            
        except Exception as e:
            splash.close()
            raise
            
    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)
        
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Startup Error",
                f"Failed to start DSSAT Viewer:\n{str(e)}"
            )
        sys.exit(1)