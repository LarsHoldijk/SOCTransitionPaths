from abc import abstractmethod

from potentials.base import Potential
from potentials.md_utils import ForceReporter


class MoleculePotential(Potential):
    """
    This is a baseclass for implementing molecular potentials using some framework.
    """

    def __init__(self, start_file, index, save_file=None):
        super().__init__()
        self.start_file = start_file  # Path to the startign conformation of paths
        self.index = index  # Multiple OpenMM runs are managed for each molecule. This is the index for keeping track
        self.save_file = save_file  # Location of the intermediate steps

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
        """
        Initializes all Framework specific configurations
        :return pdb: Description of the atom locations
        :return simulation: Simulation environment that runs the MD
        :return external_force: Reference to custom external force implementation to be updated by control
        """
        pass

    @abstractmethod
    def get_position_file(self):
        """
        Abstact method to point towards location of intermediate files.
        :return path: Path to intermediate files
        """
        pass

    def drift(self, forces):
        """
        This is the main workhorse. This function calculates the entire dynamics update. This includes the passive
        dynamics as well as the controlled dynamics. This is done by offloading the computation step entirely to
        OpenMM through the use of a "Custom External Force".
        :param forces: Control force given by the policy
        :return: New positions and velocity of the system.
        """
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)

        new_pos = self.reporter.latest_positions.copy()
        new_forces = self.reporter.latest_forces.copy()

        return new_pos, new_forces

    def reset(self):
        """
        Resets the MD simulation to the initial minimum energy conformation.
        """
        self.simulation.loadCheckpoint(self.get_position_file())
        for i in range(len(self.pdb.positions)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)
        self.simulation.minimizeEnergy()

    def latest_position(self):
        """
        Returns the positions after the latest update step.
        :return: Current coordinates of atoms
        """
        return self.simulation.state.getPositions()
