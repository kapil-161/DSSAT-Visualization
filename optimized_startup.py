"""
Optimizations for faster application startup
This module is imported at the start of main.py to improve load time
"""
import os
import sys
import warnings
from PyQt5.QtCore import Qt, QCoreApplication

# Suppress all warnings
warnings.simplefilter("ignore")

# Application paths
def get_app_path():
    """Get the path where the application is running from."""
    return os.path.dirname(sys.executable if getattr(sys, 'frozen', False) 
                          else os.path.abspath(__file__))

# Set application path
APP_PATH = get_app_path()

# Add to Python path if needed
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

def optimize_qt_settings():
    """Configure Qt settings for faster startup and better display"""
    # Must be set before QApplication creation
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    # Additional optimizations
    os.environ.update({
        "QT_AUTO_SCREEN_SCALE_FACTOR": "0",
        "QT_SCALE_FACTOR": "1"
    })
