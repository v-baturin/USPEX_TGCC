class ElasticMLFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def youngsModulus(self, system):
        if not 'elasticML.youngsModulus' in system:
            self.utility.predictValues(system)
        return system['elasticML.youngsModulus']

    def poissonsRatio(self, system):
        if not 'elasticML.poissonsRatio' in system:
            self.utility.predictValues(system)
        return system['elasticML.poissonsRatio']

    def bulkModulus(self, system):
        if not 'elasticML.bulkModulus' in system:
            self.utility.predictValues(system)
        return system['elasticML.bulkModulus']

    def shearModulus(self, system):
        if not 'elasticML.shearModulus' in system:
            self.utility.predictValues(system)
        return system['elasticML.shearModulus']

    def pughsRatio(self, system):
        if not 'elasticML.pughsRatio' in system:
            self.utility.predictValues(system)
        return system['elasticML.pughsRatio']

    def vickersHardness(self, system):
        if not 'elasticML.vickersHardness' in system:
            self.utility.predictValues(system)
        return system['elasticML.vickersHardness']

    def fractureToughness(self, system):
        if not 'elasticML.fractureToughness' in system:
            self.utility.predictValues(system)
        return system['elasticML.fractureToughness']
