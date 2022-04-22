"""PeriFlow
"""
from periflow_sdk import manager

start_step = manager.periflow.start_step
end_step = manager.periflow.end_step
init = manager.periflow.init
train_step = manager.periflow.train_step
is_emergency_save = manager.periflow.is_emergency_save
metric = manager.periflow.metric
upload_checkpoint = manager.periflow.upload_checkpoint

__all__ = [
    'start_step',
    'end_step',
    'init',
    'train_step',
    'is_emergency_save',
    'metric',
    'upload_checkpoint',
]
