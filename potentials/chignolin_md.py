import openmm as mm
import openmm.unit as unit
from openmm import app
from openmmtools.integrators import VVVRIntegrator

from potentials.MoleculePotential import MoleculePotential


class ChignolinPotentialMD(MoleculePotential):
    def __init__(self, start_file, index, save_file=None):
        super().__init__(start_file, index, save_file)


    def setup(self):
        pdb = app.PDBFile(self.start_file)
        forcefield = app.ForceField('amber/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005
        )
        external_force = mm.CustomExternalForce("k*(fx*x + fy*y + fz*z)")

        # creating the parameters
        external_force.addGlobalParameter("k", 1000)
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = VVVRIntegrator(
            300 * unit.kelvin,  # temp
            1.0 / unit.picoseconds,  # collision rate
            1.0 * unit.femtoseconds)  # timestep

        integrator.setConstraintTolerance(0.00001)

        platform = mm.Platform.getPlatformByName('CUDA')

        properties = {'DeviceIndex': '0', 'Precision': 'mixed'}

        simulation = app.Simulation(pdb.topology, system, integrator,
                                         platform, properties)
        simulation.context.setPositions(pdb.positions)

        return pdb, simulation, external_force

    def get_position_file(self):
        return f"{self.save_file}/ChignolinPositions"