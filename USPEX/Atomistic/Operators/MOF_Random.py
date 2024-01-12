import os
import random
import shutil
from .tobacco_3.tobacco import run_tobacco_serial, __file__ as tobacco_path
TOBACCO_DIR = os.path.dirname(tobacco_path)
from .tobacco_3.configuration import CHARGES as TOBACCO_CHARGES
class MOF_Random:
    def __init__(self,utilities):
        self.cellUtility = utilities.cellUtility
        self.atomistic = utilities.atomistic
        self.MOF_utility = utilities.mofUtility
        self.junctionUtility = utilities.junctionUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.templates_path = os.path.join(TOBACCO_DIR,'templates')
        self.templates_database_path = os.path.join(TOBACCO_DIR,'template_database')
        self.edges_path = os.path.join(TOBACCO_DIR,'edges')
        self.nodes_path = os.path.join(TOBACCO_DIR,'nodes')
        self.conditions = utilities.conditions

    def __call__(self,flavourFactory = None):
        nodes, linkers = self.MOF_utility.nodes, self.MOF_utility.linkers
        for block_dir in (self.nodes_path,self.edges_path):
            block_dir_abspath = os.path.abspath(block_dir)
            for f in os.listdir(block_dir_abspath):
                os.remove(os.path.join(block_dir_abspath,f))
        self.write_building_blocks_cif(nodes)
        self.write_building_blocks_cif(linkers)
        random_templates_paths = self.MOF_utility.get_random_templates()
        MOF = self.MOF_utility.get_MOF_from_tobacco(random_templates_paths,self.cellUtility,self.atomistic)
        MOF = flavourFactory(**MOF)
        self.conditions.putConditions(MOF)
        return MOF,

    def write_building_blocks_cif(self, building_blocks):
        block_name = building_blocks[0]['name']
        blockAtomicStructure = self.simpleMoleculeUtility.molecules[block_name]
        blockSites = self.junctionUtility.molSitesMapping[block_name]
        mountPoints = []
        for s in blockSites:
            mountPoint = s.mountPoint
            mountPoints.append(mountPoint)
        self.atomistic.AtomicStructureRepresentation.writeTobaccoBuildingBlockCif(
            structure=blockAtomicStructure, filename=str(block_name),
            tobacco_path=TOBACCO_DIR, mount_point_coordinates=mountPoints
        )



