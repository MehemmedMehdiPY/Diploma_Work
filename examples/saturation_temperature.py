import sys
sys.path.append("../")
from models import SaturationTemperature

estimator = SaturationTemperature()
print(estimator(P=10.0e5))
