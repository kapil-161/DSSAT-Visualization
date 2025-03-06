import PyInstaller.__main__
import os
from pathlib import Path

# Get current directory
base_dir = Path(__file__).resolve().parent

# Define the output directory
output_dir = base_dir / "build"
os.makedirs(output_dir, exist_ok=True)

# Define path to UPX
upx_dir = r"C:\Users\kbhattarai1\Downloads\upx-5.0.0-win64\upx-5.0.0-win64"

# Define specific package imports you're using
package_includes = [
    '--hidden-import=dash',
    '--hidden-import=dash_bootstrap_components',
    '--hidden-import=pandas',
    '--hidden-import=numpy',
    '--hidden-import=plotly',
    '--hidden-import=PyQt5',
    '--hidden-import=PyQt5.QtWidgets',
    '--hidden-import=PyQt5.QtCore',
    '--hidden-import=PyQt5.QtGui',
    '--hidden-import=PyQt5.QtWebEngineWidgets',
    '--hidden-import=werkzeug.serving',
    '--hidden-import=dash.dash_table',
    '--hidden-import=plotly.graph_objects',
    '--hidden-import=plotly.subplots'
]

# Define specific excludes - but be more conservative with PyQt5
package_excludes = [
    # General exclusions
    '--exclude-module=matplotlib',
    '--exclude-module=scipy.spatial.cKDTree',
    '--exclude-module=PIL.ImageQt',
    '--exclude-module=tkinter',
    '--exclude-module=PySide2',
    '--exclude-module=IPython',
    '--exclude-module=notebook',
    '--exclude-module=pytest',
    '--exclude-module=sphinx',
    
    # PyQt5 specific exclusions (only the safe ones)
    '--exclude-module=PyQt5.QtBluetooth',
    '--exclude-module=PyQt5.QtDBus',
    '--exclude-module=PyQt5.QtDesigner',
    '--exclude-module=PyQt5.QtHelp',
    '--exclude-module=PyQt5.QtLocation',
    '--exclude-module=PyQt5.QtMultimedia',
    '--exclude-module=PyQt5.QtMultimediaWidgets',
    '--exclude-module=PyQt5.QtOpenGL',
    '--exclude-module=PyQt5.QtPositioning',
    '--exclude-module=PyQt5.QtSensors',
    '--exclude-module=PyQt5.QtSerialPort',
    '--exclude-module=PyQt5.QtSql',
    '--exclude-module=PyQt5.QtSvg',
    '--exclude-module=PyQt5.QtTest',
    '--exclude-module=PyQt5.QtXml',
    '--exclude-module=PyQt5.QtXmlPatterns'
]

# Assets to include
data_files = [
    '--add-data={0};{1}'.format(str(base_dir / 'ui' / 'assets'), 'ui/assets')
]

# Define files to exclude from UPX compression
upx_excludes = [
    '--upx-exclude=vcruntime140.dll',
    '--upx-exclude=python3*.dll',
    '--upx-exclude=VCRUNTIME140.dll',
    '--upx-exclude=api-ms-win-*.dll',
    '--upx-exclude=_uuid.pyd',
    '--upx-exclude=_ssl.pyd',
    '--upx-exclude=unicodedata.pyd',
    '--upx-exclude=_ctypes.pyd',
    '--upx-exclude=Qt5*.dll',
    '--upx-exclude=PyQt5*.dll'
]

# Run PyInstaller with optimized settings
PyInstaller.__main__.run([
    # Main script
    str(base_dir / 'main.py'),
    
    # Output settings
    '--name=DSSAT_Viewer',
    '--onedir',          # Single file executable
    '--windowed',         # No console window
    
    # Optimization flags
    '--clean',            # Clean cache before building
    
    # UPX for compression
    '--noupx',
    
    # Specific optimizations
    '--log-level=INFO',   # Show more info during build
    
    # Include/exclude packages
    *package_includes,
    *package_excludes,
    *data_files,
    *upx_excludes,
])

print("Build complete! Executable is in the dist directory.")