import warnings

message = 'You are using an old version of dypy. ' 
message += 'Please contact Nicolas Piaget for more information'
warnings.warn(message, UserWarning)
