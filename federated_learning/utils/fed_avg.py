# def average_nn_parameters(parameters):
#     """
#     Averages passed parameters.

#     :param parameters: nn model named parameters
#     :type parameters: list
#     """
#     new_params = {}
#     for name in parameters[0].keys():
#         new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

#     return new_params
import torch
def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    params = parameters[0]
    for i in range(1,len(parameters)):
        params.add_(parameters[i])
    params = torch.div(params, len(parameters))
    return params
