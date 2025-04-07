import sys
sys.path.append("../")
from models import EnthalpyPressure

model = EnthalpyPressure()

Ti = 455.4792
To = 481.4288
P=10.0e5

results = model(T=Ti, P=P, is_H2O=False)
print(results)