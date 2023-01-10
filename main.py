from PySide6.QtWidgets import QApplication
from app import Application
import sys

if __name__ == "__main__":
    application = QApplication(sys.argv)
    app = Application()
    sys.exit(application.exec())
