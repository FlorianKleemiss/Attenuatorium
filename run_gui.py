#!/usr/bin/env python
"""
Attenuatorium GUI Demo Launcher

This script launches the GUI version of the Attenuatorium tool.
"""

import sys
from attenuatorium_gui import QApplication, AttenuatoriumGUI

def main():
    app = QApplication(sys.argv)
    window = AttenuatoriumGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
