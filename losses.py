import torch
import torch.nn.functional as F

class VAE_Loss_CE:
    def __call__(self, recon_x, x, mu, logvar):
        """
        Compute the loss function for a Variational Autoencoder (VAE) with cross-entropy loss.

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
        # Reconstruction loss (Cross-entropy)
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")

        # Regularization loss (KL divergence)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = reconstruction_loss + kl_divergence

        return total_loss, reconstruction_loss, kl_divergence

class VAE_Loss_MSE:
    def __call__(self, recon_x, x, mu, logvar):
        """
        Compute the loss function for a Variational Autoencoder (VAE) with MSE loss.

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

class VAE_Loss_Patch_MSE:
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def __call__(self, recon_x, x, mu, logvar):
        """
        Compute the loss function for a Variational Autoencoder (VAE) with patch-wise MSE loss.

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
        # Extract image dimensions
        _, _, H, W = x.size()
        h_patch, w_patch = self.patch_size
        
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

class VAE_Loss_Patch_CE:
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def __call__(self, recon_x, x, mu, logvar):
        """
        Compute the loss function for a Variational Autoencoder (VAE) with patch-wise cross-entropy loss.

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
        # Extract image dimensions
        _, _, H, W = x.size()
        h_patch, w_patch = self.patch_size
        
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
