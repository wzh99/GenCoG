from ..config import config
from ..expr import iran

num_ran = iran(1, config['op.max_num'])
rank_ran = iran(config['op.min_rank'], config['op.max_rank'])
dim_ran = iran(1, config['op.max_dim'])
