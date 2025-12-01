from collections import OrderedDict
from typing import Any, Mapping, Union

from omegaconf import open_dict

import numpy as np

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
        mae_decoder: bool = False,
        patch_size: int = 4, 
        data_sizes = None,
    ):

        super().__init__()
        self.monomodal = (len(modality_models) == 1)
        self.num_species = num_species
        self.num_bins = num_bins
        self.aggregator = aggregator
        self.classifying = num_bins != -1
        self.mae_decoder = mae_decoder
        self.patch_size = patch_size
        self.data_sizes = data_sizes

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
                submodels[modality_name] = self.add_layer_norm_to_fc(submodels[modality_name])

                # Load data from checkpoint
                checkpoint = torch.load(modality_checkpoint, weights_only=False)
                state_dict = {}
                
                try:
                    for key, value in checkpoint['state_dict'].items():
                        state_dict[key.replace(f"model.modality_models.{modality_name}.", '')] = value
                    submodels[modality_name].load_state_dict(state_dict)

                except:
                    # Load from transductive checkpoint
                    submodels[modality_name], avgpool, fc = self.prepare_transductive_submodel(modality_name, submodels[modality_name])

                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_state_dict[key.replace("model.decoder", 'decoder')] = value
                    submodels[modality_name].load_state_dict(new_state_dict)

                    del(submodels[modality_name].decoder)
                    submodels[modality_name].avgpool = avgpool
                    submodels[modality_name].fc = fc



            else:
                submodels[modality_name] = check_model(model)
                submodels[modality_name] = self.add_layer_norm_to_fc(submodels[modality_name])
                

        self.modality_models = nn.ModuleDict(submodels)


        # Add decoders if running in MAE mode
        if self.mae_decoder and not(hasattr(self, 'decoder')):
            self.create_decoder()
            

        # Prepare aggregation
        if not self.monomodal and not self.mae_decoder:

            # Extract fc layers from submodels
            linears = []

            for modality_name in self.modality_models:

                linears.append(self.pop_last_linear(modality_name))

                # Freeze submodels
                if freeze_submodels:
                    self.modality_models[modality_name].eval()

            # Initialize aggregator model
            if aggregator == 'Linear':

                # Use weights from submodels' linear layers stacked together
                
                lin = nn.Linear(sum([x.in_features for x in linears]), linears[0].out_features)

                with torch.no_grad():
                    lin.weight.data = torch.cat([li.weight.data for li in linears], dim=1) / len(linears)
                    lin.bias.data = torch.mean(torch.stack([li.bias.data for li in linears]), dim=0)

                self.aggregator_model = nn.Sequential(nn.LayerNorm(lin.in_features), lin)
            
            elif aggregator == 'MLP':

                self.aggregator_model =  torchvision.ops.MLP(in_channels = sum([x.in_features for x in linears]),
                                                        hidden_channels = [linears[0].out_features // 4, linears[0].out_features],
                                                        dropout = 0.3)


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


        if self.mae_decoder:
            x = x['masked_patches']
            outputdict = {}
            inputs = []

            for modality_name, model in self.modality_models.items():
                out = model(x[modality_name])
                out = out.to(next(self.decoder.parameters()).device)
                inputs.append(out.view(out.shape[0], -1))

            output = self.decoder(torch.concat(inputs, dim=1))

            for mod in self.modality_models:
                ix = np.prod(self.data_sizes[mod])
                outputdict[mod], output = output[...,:ix], output[...,ix:]

            return {mod: outputdict[mod].view(-1, *self.data_sizes[mod]) for mod in outputdict}


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
                return out.view(out.shape[:-1] + (self.num_species, self.num_bins))

        else:
            return out


    def add_layer_norm_to_fc(self, modality: nn.Module) -> nn.Module:

        # Add LayerNorm to last linear layer

        _, layername = _find_module_of_type(modality, nn.Linear, 'last')
        in_size = getattr(modality, layername).in_features
        new_layer = nn.Sequential(nn.LayerNorm(in_size), getattr(modality, layername))
        setattr(modality, layername, new_layer)

        return modality


    def pop_last_linear(self, modality_name: str) -> nn.Module:

        # Remove linear and return it, in order to concatenate multiple fc layers

        _, layername = _find_module_of_type(self.modality_models[modality_name], nn.Sequential, 'last')

        lin = getattr(self.modality_models[modality_name], layername)[1]
        getattr(self.modality_models[modality_name], layername)[1] = nn.Identity()

        return lin


    def create_decoder(self):

        # Create decoders for MAE model

        layerinput = 0
        outsize = 0

        for mod in self.modality_models:

            layerinput += self.data_sizes[mod][-2] * self.data_sizes[mod][-1]
            outsize += np.prod(self.data_sizes[mod])

            self.modality_models[mod].avgpool = nn.Identity()
            self.modality_models[mod].fc = nn.Identity()

        self.decoder = nn.Sequential(
                nn.Linear(layerinput // 2, outsize // 8),
                nn.GELU(),
                nn.Linear(outsize // 8, outsize)
            )


    def remove_final_layer(self):
        """Remove the final layers of the model to keep only the feature extractor."""

        self.aggregator_model[1] = nn.Identity()


    def edit_final_layer(self, new_species_num):
        """Edit the final layer of the model to change the number of output classes."""

        self.aggregator_model[1] = nn.Linear(self.aggregator_model[1].in_features, new_species_num * self.num_bins)


    def pop_last_layers(self):
        """Remove last layers to be able to load transductive checkpoint"""

        aggregator = self.aggregator_model
        self.aggregator_model = nn.Identity()
        
        avgpool, fc = {}, {}
        layerinput, outsize = 0, 0
        
        for mod in self.modality_models:
            avgpool[mod] = self.modality_models[mod].avgpool
            fc[mod] = self.modality_models[mod].fc

            self.modality_models[mod].avgpool = nn.Identity()
            self.modality_models[mod].fc = nn.Identity()


            layerinput += self.data_sizes[mod][-2] * self.data_sizes[mod][-1]
            outsize += np.prod(self.data_sizes[mod])
            
        self.decoder = nn.Sequential(
                nn.Linear(layerinput // 2, outsize // 8),
                nn.GELU(),
                nn.Linear(outsize // 8, outsize)
            )
            
        return aggregator, avgpool, fc
    

    def set_last_layers(self, aggregator, avgpool, fc):
        """Set back the last layers after loading transductive checkpoint."""
        
        self.aggregator_model = aggregator
        del(self.decoder)

        for mod in self.modality_models:

            self.modality_models[mod].avgpool = avgpool[mod]
            self.modality_models[mod].fc = fc[mod]


    def prepare_transductive_submodel(self, modality, submodel):
        """Edit submodel structure to be able to load transductive checkpoint."""
        
        layerinput = self.data_sizes[modality][-2] * self.data_sizes[modality][-1]
        outsize = np.prod(self.data_sizes[modality])
        
        avgpool, fc = submodel.avgpool, submodel.fc

        submodel.avgpool = nn.Identity()
        submodel.fc = nn.Identity()
        submodel.decoder = nn.Sequential(
                nn.Linear(layerinput // 2, outsize // 8),
                nn.GELU(),
                nn.Linear(outsize // 8, outsize)
            )
        
        return submodel, avgpool, fc
