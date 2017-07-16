
# Requires BigML Python bindings
#
# Install via: pip install bigml
#
# or clone it:
#   git clone https://github.com/bigmlcom/python.git

from bigml.model import Model
from bigml.api import BigML

# Downloads and generates a local version of the model, if it
# hasn't been downloaded previously.

model = Model('model/5900dbaf014404467d000811',
              api=BigML("jaimevalero78",
                        "6d685bf8cd3873a510b86500895071bcdd3d0990",
                        dev_mode=True,
                        domain="bigml.io"))

# To make predictions fill the desired input_data
# (e.g. {"petal length": 1, "sepal length": 3})
# as first parameter in next line.
model.predict({}, with_confidence=True)

# The result is a list of three elements: prediction, confidence and
# distribution

