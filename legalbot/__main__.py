"""
Entry point for running legalbot as a module: python -m legalbot
"""

from legalbot.cli.commands import app

if __name__ == "__main__":
    app()
