#!/usr/bin/env python
"""
Attenuatorium GUI Demo Launcher

This script launches the GUI version of the Attenuatorium tool.
"""

import sys
from attenuatorium_gui import QApplication, AttenuatoriumGUI

def main():
    """
    Initialize and run the Attenuatorium GUI application.
    
    This function creates the main application window, displays it,
    and starts the Qt event loop. The application will continue running
    until the user closes the window or the application is terminated.
    
    Returns:
        None: This function does not return as it calls sys.exit()
    """
    app = QApplication(sys.argv)
    window = AttenuatoriumGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
