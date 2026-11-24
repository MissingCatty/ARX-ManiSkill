"""Resident pilot services with a checked continuous terminal corridor."""
from pathlib import Path
import runpy
import arx_direct_terminal_20261120 as terminal

terminal.install()
runpy.run_path(str(Path(__file__).with_name('arx_direct_services_20261120.py')), run_name='__main__')
