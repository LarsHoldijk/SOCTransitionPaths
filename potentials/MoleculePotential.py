from abc import abstractmethod
from potentials.base import Potential
from potentials.md_utils import ForceReporter


class MoleculePotential(Potential):
    def __init__(self, start_file, index, reset_steps=100, save_file=None, temp_mul=1):
        super().__init__()
        self.start_file = start_file
        self.reset_steps = reset_steps
        self.index = index
        self.save_file = save_file
        self.temp_mul = temp_mul
        self.pdb, self.simulation, self.external_force = self.setup()

        self.reporter = ForceReporter(1)
        self.simulation.reporters.append(self.reporter)

        if self.index == 0:
            self.simulation.step(1)
            self.simulation.minimizeEnergy()
            self.simulation.saveCheckpoint(self.get_position_file())

        if self.index > -1:
            self.reset()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_position_file(self):
        pass

    def drift(self, forces):
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)

        new_pos = self.reporter.latest_positions.copy()
        new_forces = self.reporter.latest_forces.copy()

        return new_pos, new_forces

    def reset(self):
        self.simulation.loadCheckpoint(self.get_position_file())
        for i in range(len(self.pdb.positions)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)
        self.simulation.minimizeEnergy()

    def latest_position(self):
        return self.simulation.state.getPositions()
