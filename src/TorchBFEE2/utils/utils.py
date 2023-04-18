import torch
from ..models.quaternion import Quaternion


def save_quaternion(groupA_idxs, refpos, save_path='quaternion.pt'):

    model = Quaternion(groupA_idxs, refpos)
    script_module = torch.jit.script(model)
    script_module = torch.jit.freeze(script_module.eval())
    script_module.save(save_path)

    # try:
    #     script_module = torch.jit.script(model)
    #     script_module = torch.jit.freeze(script_module.eval())
    #     script_module.save(save_path)
    #     print("The model saved successfully")
    # except:

    #     print("Can not save the model")
