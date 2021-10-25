# SPDX-License-Identifier: MIT
# Source: https://github.com/microsoft/MaskFlownet/tree/5cba12772e2201f0d1c1e27161d224e585334571
from . import layer
from . import MaskFlownet
from . import pipeline

def get_pipeline(network, **kwargs):
	if network == 'MaskFlownet':
		return pipeline.PipelineFlownet(**kwargs)
	else:
		raise NotImplementedError
