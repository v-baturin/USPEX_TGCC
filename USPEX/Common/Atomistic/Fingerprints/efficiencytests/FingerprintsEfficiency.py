from lib.Fingerprints.Fingerprints import Fingerprints
from lib.Systems.Crystal.CrystalConfig import CrystalConfig
from lib.Systems.Crystal.RandTop import RandTop

import time


config = CrystalConfig(symbols=['Mg', 'Al', 'O'], blocks=[[4, 8, 16]], fixed=[[1,1]],
                       minAngle = 55, minDiagAngle = 30, minVectorLength = 1,
                       externalPressure=100, fingerprints={})
randtop = RandTop(config, initFrac=1.0,minFrac=1.0)
randtop.prepare()
system = randtop()
randtop.standby()

start = time.time()

for i in range(100):
    f = Fingerprints(system,**config.fingerprints)

end = time.time()

print (end - start)