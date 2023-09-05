import re
import torch
import config as cfg

def reward_fn(response: str, correct_answer: str) -> torch.Tensor:
    # Check for correct format

    # Normally you would use an f-string here, but it messes with some of the 
    # raw string formatting and is kind of a pain
    valid_format = (
            r'[!"#$%&\\\'()*+,\-./:;?@[\\\]\^_`{\|}~]{' 
            + str(cfg.NUM_STEG_TOKENS) 
            + r',}\nANS: \d+'
        )
    if not re.match(valid_format, response):
        return torch.tensor(-1.)

    # Check for correct answer
    response_answer = re.match(r'\d+', response)
    if (response_answer is None) or \
        (response_answer.group() != correct_answer):
        return torch.tensor(-1.)
    
        
    return torch.tensor(1.)
