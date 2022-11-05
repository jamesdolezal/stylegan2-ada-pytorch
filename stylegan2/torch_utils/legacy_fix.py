def apply_legacy_patch(src):
    src = src.replace(
        '''from ..torch_utils import misc, persistence
from ..torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d''',
        '''try:
    from ..torch_utils import misc, persistence
    from ..torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d
except ImportError:
    from torch_utils import misc, persistence
    from torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d'''
    )
    src = src.replace(
        '''from ..torch_utils import misc, persistence
from ..torch_utils.ops import conv2d_gradfix, grid_sample_gradfix, upfirdn2d''',
        '''try:
    from ..torch_utils import misc, persistence
    from ..torch_utils.ops import conv2d_gradfix, grid_sample_gradfix, upfirdn2d
except ImportError:
    from torch_utils import misc, persistence
    from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix, upfirdn2d'''
    )
    return src