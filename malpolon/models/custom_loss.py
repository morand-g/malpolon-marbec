import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LogSpacingLoss(nn.modules.loss._Loss):

    def __init__(self, num_bins, num_species, loss_weights=None):
        super(LogSpacingLoss, self).__init__()

        self.num_bins = num_bins
        self.num_species = num_species
        self.loss_weights = torch.Tensor(loss_weights) if loss_weights is not None else None

        # Precompute log-scaled distances
        log_indices = torch.log1p(torch.arange(num_bins, dtype=torch.float32))
        distances = torch.abs(log_indices.unsqueeze(0) - log_indices.unsqueeze(1))
        distances = torch.exp(distances)
        self.register_buffer("distances", distances)  # Store as a non-trainable tensor


    def forward(self, predictions, target):

        target_indices = target.to(torch.int64)

        prob_predictions = torch.softmax(predictions, dim=-1)
        calculated_distances = self.distances[target_indices]
        loss = (calculated_distances * prob_predictions)

        if self.loss_weights is not None:
            loss = (loss * self.loss_weights.to(loss.device)).mean(dim=-1)

        return loss.mean()


class ModifiedCELoss(nn.modules.loss._Loss):

    def __init__(self, num_bins, num_species, mse_alpha = 0, loss_weights=None):
        super(ModifiedCELoss, self).__init__()

        self.num_bins = num_bins
        self.num_species = num_species
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32) if loss_weights is not None else None
        
        self.mse_alpha = mse_alpha
        if mse_alpha > 0:
            self.mse_loss = ClassifMSELoss()

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_species, num_classes) -> Raw logits
        targets: (batch_size, num_species) -> Class indices (0 to num_classes - 1)
        """

        # Reshape for cross-entropy compatibility
        predictions = predictions.view(-1, self.num_bins)  # (batch_size * num_species, num_classes)
        targets = targets.view(-1).to(torch.int64)  # (batch_size * num_species,)

        # Apply optional class weights
        if self.loss_weights is not None:
            loss = F.cross_entropy(predictions, targets, weight=self.loss_weights.to(predictions.device), reduction='mean')
        else:
            loss = F.cross_entropy(predictions, targets, reduction='mean')

        if mse_alpha > 0:
            loss = (1 - self.mse_alpha) * loss + self.mse_alpha * self.mse_loss(predictions, targets)
            
        return loss


class SumMSELoss(nn.modules.loss._Loss):
    def __init__(self, **kwargs):
        """
        Loss function for Masked Autoencoding

        Parameters
        ----------
        patch_size : int
            Size of each patch (e.g., 4 for 4x4 patches).
        num_layers : int
            Number of layers in the input (e.g., 19).
        """
        super(SumMSELoss, self).__init__()

    def forward(self, predictions, targets) -> torch.Tensor:
        """
        Compute the MSE loss for Masked Auto Encoding

        Parameters
        ----------
        reconstructed_patches : torch.Tensor
            Reconstructed patches from the model. Shape: (batch_size, num_patches, num_layers, patch_size, patch_size).
        original_patches : torch.Tensor
            Original patches from the input. Shape: (batch_size, num_patches, num_layers, patch_size, patch_size).
        mask : torch.Tensor
            Binary mask indicating which patches were masked. Shape: (batch_size, num_patches).

        Returns
        -------
        torch.Tensor
            Computed MSE loss.
        """

        loss = 0
        
        for mod in targets:
            reconstructed_patches = predictions[mod].view(targets[mod].shape[0], -1)
            loss += F.mse_loss( reconstructed_patches,
                                targets[mod].view(targets[mod].shape[0], -1),
                                reduction='mean')
        
        return loss  
    

class CeSrLoss(nn.modules.loss._Loss):

    def __init__(self, num_bins, num_species, loss_weights=None, alpha = 0.01):
        super(CeSrLoss, self).__init__()

        self.num_bins = num_bins
        self.num_species = num_species
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32) if loss_weights is not None else None
        self.alpha = alpha  # Weight for the species richness term

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_species, num_classes) -> Raw logits
        targets: (batch_size, num_species) -> Class indices (0 to num_classes - 1)
        """

        # Baseline :  predicted_sr =  torch.log(1+(predictions[...,1] > predictions[...,0]).sum(axis=1))
        # Test 1 :  predicted_sr =  torch.log(1+(predictions[...,1] < predictions[...,0]).sum(axis=1))
        # Test 2 : 
        probas = torch.softmax(predictions, dim=-1)
        predicted_sr = torch.log(1+(probas[...,1]).sum(axis=1))

        target_sr = torch.log(1+(targets > 0).sum(axis=1))
        sr_term = F.mse_loss(predicted_sr, target_sr, reduction='mean')

        # Reshape for cross-entropy compatibility
        predictions = predictions.view(-1, self.num_bins)  # (batch_size * num_species, num_classes)
        targets = targets.view(-1).to(torch.int64)  # (batch_size * num_species,)

        # Apply optional class weights
        if self.loss_weights is not None:
            loss = F.cross_entropy(predictions, targets, weight=self.loss_weights.to(predictions.device), reduction='mean')
        else:
            loss = F.cross_entropy(predictions, targets, reduction='mean')

        return loss + self.alpha * sr_term

    

class FilteredHuberLoss(nn.HuberLoss):

    def __init__(self, delta: float = 1.0) -> None:

        super().__init__(delta=delta, reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        present = (target != 0).to(int)
        huber_loss = super().forward(input, target)
        filtered_loss = (huber_loss * present).mean()
        return filtered_loss



class FilteredMSELoss(nn.MSELoss):

    def __init__(self) -> None:

        super().__init__(reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        present = (target != 0).to(int)
        mse_loss = super().forward(input, target)
        filtered_loss = (mse_loss * present).mean()
        return filtered_loss

    
class ClassifMSELoss(nn.MSELoss):

    def __init__(self) -> None:

        super().__init__()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:

        bin_indices = torch.arange(preds.shape[-1], device=preds.device).float()  # Shape: (num_bins,)
        expected_bin = torch.sum(torch.softmax(preds, dim=-1) * bin_indices, dim=-1)  # Shape: (batch_size, num_bins)
        mse_loss = super().forward(expected_bin, targets)

        return mse_loss