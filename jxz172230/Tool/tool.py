import torch
import torch.nn as nn
import csv
from torch.autograd import Variable
import re
from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device="cuda"):

    def display(model, memory_size=6000000, bandwidth=5120000000, frequency=1/(10**9), mac_unit=0:


        # outputWrite = csv.writer(open('CNN_Tool.csv', 'wb'), delimiter=' ', quotechar='|',
        #            quoting=csv.QUOTE_MINIMAL)

        title = "{:>50}".format("Your Model:    ") + model.__class__.__name__
        print(title)

        print()


        singleline = "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        print(singleline)

        line_new = "{:>20} {:>25} {:>25} {:>25} {:>15} {:>10} {:>5} {:>10} {:>5} {:>10} {:>10} {:>20} {:>10} {:>10} {:>5} {:>15} {15}"\
            .format("Layer (type)", "Input(BS, Ni, Lr, Lc)", "Filter(No, Ni, Fr, Fc)", "Output(BS, No, Mr, Mc)", "Sampling(Sr, Sc)",
                    "In_Mem","In_Move", "Out_Mem", "Out_Move","Band", "Band_time", "BLAS(M, N, K)", "MACs", "Cx_Mem", "Cx_Move", "Param #")
        print(line_new)

        doubleline = "================================================================================================================================================================================================================================================================"
        print(doubleline)

        total_params = 0
        total_output = 0
        trainable_params = 0
        index = 0
        for layer in summary:
            index += 1
            ni = summary[layer]["input_shape"][1]
            lr = summary[layer]["input_shape"][2] if len(summary[layer]["input_shape"]) > 2 else 1
            lc = summary[layer]["input_shape"][3] if len(summary[layer]["input_shape"]) > 2 else 1
            no = summary[layer]["output_shape"][1]
            mr = summary[layer]["output_shape"][2] if len(summary[layer]["output_shape"]) > 2 else 1
            mc = summary[layer]["output_shape"][3] if len(summary[layer]["output_shape"]) > 2 else 1
                if summary[layer]["filter"] == None:
                    fr = None
                    fc = None
                elif isinstance(summary[layer]["filter"], int):
                    fr = summary[layer]["filter"] if summary[layer]["filter"] != None else None
                    fc = 1
                elif len(summary[layer]["filter"]) == 2:
                    fr = summary[layer]["filter"][0]
                    fc = summary[layer]["filter"][1]
            in_memory = ni * lr * lc
            out_memory = no * mr * mc
            in_move = in_memory // memory_size
            out_move = out_memory // memory_size
            cx_memory = 0 if summary[layer]["class"] == 'Else' else no * ni * lr * lc
            cx_move = 0 if summary[layer]["class"] == 'Else' else 1
            band = in_move * in_memory + out_memory * out_memory
            time_band = band / bandwidth

            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20} {:>25} {:>25} {:>25} {:>15} {:>10} {:>5} {:>10} {:>5} {:>10} {:>10} {:>20} {:>15} {:>5} {:>10}  {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                str([no, ni, fr,fc]) if fr or fc else "None",
                str(summary[layer]["output_shape"]),
                str(summary[layer]["sampling"]),
                str(in_memory),
                str(in_move),
                str(out_memory),
                str(out_move),
                str(band),
                str(time_band),
                str(summary[layer]["blas"]),
                str(summary[layer]["MACs"]),
                str(cx_memory),
                str(cx_move),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)
            ###################

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print(doubleline)
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print(singleline)
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print(singleline)
        # return summary




    def register_hook(module):

        ##################################################################
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            # input shape
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            # output shape
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            summary[m_key]["sampling"] = module.stride if hasattr(module, "stride") else None
            # BLAS build and MAC build and filter build
            if re.search('con', class_name, re.IGNORECASE):
                kernelsize = module.kernel_size if hasattr(module, "kernel_size") else module.conv.kernel_size
                summary[m_key]["class"] = "Conv"
                summary[m_key]["blas"] = [summary[m_key]["output_shape"][1], summary[m_key]["output_shape"][2] * summary[m_key]["output_shape"][3],\
                                          summary[m_key]["input_shape"][1] * kernelsize[0] * kernelsize[1]]
                summary[m_key]["MACs"] = summary[m_key]["output_shape"][1] * summary[m_key]["output_shape"][2] * summary[m_key]["output_shape"][3]\
                                         * kernelsize[0] * kernelsize[1]
                summary[m_key]["filter"] = kernelsize
            elif re.search('linear', class_name, re.IGNORECASE):
                summary[m_key]["class"] = "Linear"
                summary[m_key]["blas"] = [summary[m_key]["output_shape"][1], 1, summary[m_key]["input_shape"][1]]
                summary[m_key]["MACs"] = summary[m_key]["output_shape"][1] * summary[m_key]["input_shape"][1]
                summary[m_key]["filter"] = [1,1]
            elif re.search('pool', class_name, re.IGNORECASE):
                summary[m_key]["class"] = "Pooling"
                summary[m_key]["blas"] = None
                summary[m_key]["MACs"] = None
                summary[m_key]["filter"] = module.kernel_size
                summary[m_key]["sampling"] = summary[m_key]["input_shape"]
            else:
                summary[m_key]["class"] = "Else"
                summary[m_key]["blas"] = [summary[m_key]["output_shape"][1], summary[m_key]["output_shape"][2] * summary[m_key]["output_shape"][3], \
                                          summary[m_key]["input_shape"][1] * module.kernel_size[0] * module.kernel_size[1]] if hasattr(module, "kernel_size") else None
                summary[m_key]["MACs"] = summary[m_key]["output_shape"][1] * summary[m_key]["output_shape"][2] * summary[m_key]["output_shape"][3]\
                                         * module.kernel_size[0] * module.kernel_size[1] if hasattr(module, "kernel_size") else None
                summary[m_key]["filter"] = [1,1]
            # the parameter we may needs
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))


    ####################################################################
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    # dtype - device type; it will take consider of all possible combination of a tuple
    # * :unpack tuple, if tuple has 4 elements, then total 24 combination
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    display(model)


