"""SepsisTRACE desktop app: early sepsis warning on ICU data (PhysioNet/CinC 2019).

Run from this folder:   python main.py
"""
from gui.app import SepsisApp


def main() -> None:
    SepsisApp().mainloop()


if __name__ == "__main__":
    main()
