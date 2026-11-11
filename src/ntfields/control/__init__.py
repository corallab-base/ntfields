"""Online control on a trained field: the controller reads the field at the current state, so it closes the loop
around tracking error and disturbances, where a planner's path would only be tracked."""

from ntfields.control.dynamics import Dynamics, PinocchioDynamics, PointMassDynamics
from ntfields.control.geodesic_tau import GeodesicTauConfig, GeodesicTauController

__all__ = ["Dynamics", "GeodesicTauConfig", "GeodesicTauController", "PinocchioDynamics", "PointMassDynamics"]
