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

def VAE_Loss_Patch_MSE(recon_x, x, mu, logvar, patch_size):
    """
    Compute the loss function for a Variational Autoencoder (VAE) with patch-wise MSE loss.

    Args:
        recon_x (torch.Tensor): Reconstructed output from the decoder.
        x (torch.Tensor): Input data.
        mu (torch.Tensor): Mean of the latent space.
        logvar (torch.Tensor): Log variance of the latent space.
        patch_size (tuple): Size of the patches (height, width).

    Returns:
        torch.Tensor: Total loss.
        torch.Tensor: Reconstruction loss.
        torch.Tensor: Regularization loss (KL divergence).
    """
    # Patch-wise MSE loss
    # Extract image dimensions
    _, _, H, W = x.size()
    h_patch, w_patch = patch_size
    
    # Calculate the number of patches
    num_patches_h = H // h_patch
    num_patches_w = W // w_patch
    
    # Reshape recon_x and x into patches
    recon_x_patches = recon_x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    x_patches = x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    
    # Calculate MSE loss for each patch
    patch_losses = F.mse_loss(recon_x_patches, x_patches, reduction='none')
    
    # Compute mean MSE loss across all patches
    reconstruction_loss = patch_losses.mean(dim=(2, 3, 4))

    # Regularization loss (KL divergence)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss.mean() + kl_divergence

    return total_loss, reconstruction_loss.mean(), kl_divergence

def VAE_Loss_Patch_CE(recon_x, x, mu, logvar, patch_size):
    """
    Compute the loss function for a Variational Autoencoder (VAE) with patch-wise cross-entropy loss.

    Args:
        recon_x (torch.Tensor): Reconstructed output from the decoder.
        x (torch.Tensor): Input data.
        mu (torch.Tensor): Mean of the latent space.
        logvar (torch.Tensor): Log variance of the latent space.
        patch_size (tuple): Size of the patches (height, width).

    Returns:
        torch.Tensor: Total loss.
        torch.Tensor: Reconstruction loss.
        torch.Tensor: Regularization loss (KL divergence).
    """
    # Extract image dimensions
    _, _, H, W = x.size()
    h_patch, w_patch = patch_size
    
    # Calculate the number of patches
    num_patches_h = H // h_patch
    num_patches_w = W // w_patch
    
    # Reshape recon_x and x into patches
    recon_x_patches = recon_x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    x_patches = x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    
    # Calculate cross-entropy loss for each patch
    patch_losses = F.binary_cross_entropy(recon_x_patches, x_patches, reduction='none')
    
    # Compute mean cross-entropy loss across all patches
    reconstruction_loss = patch_losses.mean(dim=(2, 3, 4))
    
    # Regularization loss (KL divergence)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss.mean() + kl_divergence

    return total_loss, reconstruction_loss.mean(), kl_divergence
