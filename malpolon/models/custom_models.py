from collections import OrderedDict
from typing import Any, Mapping, Union

from omegaconf import open_dict

import torch
from torch import nn
import torchvision

from .utils import check_model
from .model_builder import _find_module_of_type


class MultiModalModel(nn.Module):

    def __init__(
        self,
        modality_models: Union[nn.Module, Mapping],
        num_species,
        num_bins,
        aggregator: str,
        freeze_submodels: bool = False,
    ):

        super().__init__()
        self.monomodal = (len(modality_models) == 1)
        self.num_species = num_species
        self.num_bins = num_bins
        self.aggregator = aggregator
        self.classifying = True

        if num_bins == -1:
            self.classifying = False

        # Load submodels from checkpoint or from scratch

        submodels = OrderedDict()

        for modality_name, model in modality_models.items():

            if 'checkpoint' in model and model['checkpoint'] is None:

                # Remove empty checkpoint field
                with open_dict(model):
                    del model['checkpoint']

            if 'checkpoint' in model:

                # Read checkpoint path and delete it from dict
                modality_checkpoint = model['checkpoint']
                model_copy = model.copy()

                with open_dict(model_copy):
                    del model_copy['checkpoint']

                # Instantiate model
                submodels[modality_name] = check_model(model_copy)
            
                # Add LayerNorm to last linear layer to be able to load checkpoint
                submodels[modality_name] = self.add_layer_norm(submodels[modality_name])

                # Load data from checkpoint
                checkpoint = torch.load(modality_checkpoint, weights_only=False)
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    state_dict[key.replace(f"model.modality_models.{modality_name}.", '')] = value
                submodels[modality_name].load_state_dict(state_dict)

            else:
                submodels[modality_name] = check_model(model)
                submodels[modality_name] = self.add_layer_norm(submodels[modality_name])
                

        self.modality_models = nn.ModuleDict(submodels)

        # Prepare aggregation
        if not self.monomodal:

            # Extract fc layers from submodels
            linears = []

            for modality_name in self.modality_models:

                linears.append(self.pop_linear(modality_name))

                # Freeze submodels
                if freeze_submodels:
                    self.modality_models[modality_name].eval()

            # Initialize aggregator model with extracted weights

            if aggregator == 'Linear':
                
                lin = nn.Linear(sum([x.in_features for x in linears]), linears[0].out_features)

                with torch.no_grad():
                    lin.weight.data = torch.cat([li.weight.data for li in linears], dim=1) / len(linears)
                    lin.bias.data = torch.mean(torch.stack([li.bias.data for li in linears]), dim=0)

                self.aggregator_model = nn.Sequential(nn.LayerNorm(lin.in_features), lin)
            
            elif aggregator == 'MLP':

                self.aggregator_model =  torchvision.ops.MLP(in_channels = sum([x.in_features for x in linears]),
                                                        hidden_channels = [linears[0].out_features // 4, linears[0].out_features],
                                                        dropout = 0.1)


    def forward(self, *arg) -> Any:

        # Check if input is a dict or a tuple
        if type(arg[0]) != dict:
            mods = list(self.modality_models.keys())
            if len(arg) == len(mods):
                x = {mods[i]:arg[i] for i in range(len(arg))}
            else:
                raise ValueError(f"The number of inputs ({len(arg)}) does not match the number of submodels ({len(self.modality_models)}).")
        else:
            x = arg[0]

        if self.monomodal:
            modname = list(self.modality_models.keys())[0]
            modality = self.modality_models[modname]
            out = modality(x[modname])

        else:
            features = []

            for modality_name, model in self.modality_models.items():

                out = model(x[modality_name])
                out = out.to(next(self.aggregator_model.parameters()).device)
                features.append(out)

            features = torch.concat(features, dim=-1)
            out = self.aggregator_model(features)

        if self.classifying:
            # out_probs = torch.softmax(out, dim=-1)
            # out_probs = out_probs.view(out_probs.shape[:-2]+ (self.num_species * self.num_bins,))
            return out.view(out.shape[:-1] + (self.num_species, self.num_bins))

        else:
            return out


    def add_layer_norm(self, modality: nn.Module) -> nn.Module:

        # Add LayerNorm to last linear layer

        _, layername = _find_module_of_type(modality, nn.Linear, 'last')
        in_size = getattr(modality, layername).in_features
        new_layer = nn.Sequential(nn.LayerNorm(in_size), getattr(modality, layername))
        setattr(modality, layername, new_layer)

        return modality


    def pop_linear(self, modality_name: str) -> nn.Module:

        # Remove linear and return it

        _, layername = _find_module_of_type(self.modality_models[modality_name], nn.Sequential, 'last')

        lin = getattr(self.modality_models[modality_name], layername)[1]
        getattr(self.modality_models[modality_name], layername)[1] = nn.Identity()

        return lin