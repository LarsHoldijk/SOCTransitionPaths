import numpy as np
import openmm.unit as unit


class ForceReporter(object):
    def __init__(self, reportInterval):
        self._reportInterval = reportInterval
        self.force_history = []
        self.position_history = []
        self.potontial_history = []

        self.max_potential = -np.inf
        self.max_potential_step = 0
        self.step = 0

    def __del__(self):
        return

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, True, True, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        self.latest_forces = np.asarray(forces)
        
        pos = state.getPositions().value_in_unit(unit.nanometer)
        self.latest_positions = np.asarray(pos)
        
        pot = state.getPotentialEnergy().value_in_unit(unit.kilojoules/unit.mole)
        self.latest_potential = np.asarray(pot)

        if self.latest_potential > self.max_potential:
            self.max_potential_step = self.step
            self.max_potential = self.latest_potential

        self.step += 1

