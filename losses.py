import torch
import torch.nn.functional as F

def VAE_Loss(recon_x, x, mu, logvar):
    """
    Compute the loss function for a Variational Autoencoder (VAE).

    Args:
        recon_x (torch.Tensor): Reconstructed output from the decoder.
        x (torch.Tensor): Input data.
        mu (torch.Tensor): Mean of the latent space.
        logvar (torch.Tensor): Log variance of the latent space.

    Returns:
        torch.Tensor: Total loss.
        torch.Tensor: Reconstruction loss.
        torch.Tensor: Regularization loss (KL divergence).
    """
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(recon_x, x, reduction="mean")
    
    # Check if reconstruction loss is NaN
    if torch.isnan(reconstruction_loss):
        # Handle NaN value (e.g., replace with a default value)
        reconstruction_loss = torch.tensor(0.0)  # Replace NaN with 0.0 or any other appropriate value
    
    # Regularization loss (KL divergence)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss + kl_divergence

    return total_loss, reconstruction_loss, kl_divergence
