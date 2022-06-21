
from openmm import app
import openmm as mm
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator
import numpy as np 
from tqdm import tqdm

from potentials.MoleculePotential import MoleculePotential

class AlaninePotentialMD(MoleculePotential):
    def __init__(self, start_file, index, reset_steps=100, save_file=None, temp_mul = 1):
        super().__init__(start_file, index, reset_steps, save_file, temp_mul)

    def setup(self):
        # print(self.temp_mul)
        forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
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
            300 * unit.kelvin * self.temp_mul,  # temp
            1.0 / unit.picoseconds,  # collision rate
            1.0 * unit.femtoseconds)  # timestep

        integrator.setConstraintTolerance(0.00001)

        platform = mm.Platform.getPlatformByName('CUDA')

        properties = {'DeviceIndex': '0', 'Precision': 'mixed'}

        simulation = app.Simulation(pdb.topology, system, integrator,
                                         platform, properties)
        simulation.context.setPositions(pdb.positions)

        # not sure if you have to report every step, but if you don't, it might be better to increase it
        # (for example to 10, because that is how many steps you take in the method potential)
        # it is going to be slower when it has to report every step, but

        return pdb, simulation, external_force

    def get_position_file(self):
        return f"{self.save_file}/AlaninePositions"