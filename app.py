"""Streamlit Community Cloud entry point for the dashboard."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


# Do not use ``import app`` here: it is ambiguous with this entry-point file
# and can silently import the wrong module as the project grows.
dashboard_path = Path(__file__).parent / "ai" / "app.py"
spec = spec_from_file_location("zomato_dashboard", dashboard_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load dashboard from {dashboard_path}")

dashboard = module_from_spec(spec)
sys.modules[spec.name] = dashboard
spec.loader.exec_module(dashboard)
