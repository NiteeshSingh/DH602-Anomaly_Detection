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
        # if torch.isnan(reconstruction_loss):
        #     # Handle NaN value (e.g., replace with a default value)
        #     reconstruction_loss = torch.tensor(0.0)  # Replace NaN with 0.0 or any other appropriate value

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
        recon_x_patches = recon_x.unfold(2, h_patch, h_patch).unfold(
            3, w_patch, w_patch
        )
        x_patches = x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)

        # Calculate MSE loss for each patch
        patch_losses = F.mse_loss(recon_x_patches, x_patches, reduction="none")

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
        recon_x_patches = recon_x.unfold(2, h_patch, h_patch).unfold(
            3, w_patch, w_patch
        )
        x_patches = x.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)

        # Calculate cross-entropy loss for each patch
        patch_losses = F.binary_cross_entropy(
            recon_x_patches, x_patches, reduction="none"
        )

        # Compute mean cross-entropy loss across all patches
        reconstruction_loss = patch_losses.mean(dim=(2, 3, 4))

        # Regularization loss (KL divergence)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = reconstruction_loss.mean() + kl_divergence

        return total_loss, reconstruction_loss.mean(), kl_divergence


class RA_Loss:
    def __init__(self, loss_type="bce"):
        self.loss_type = loss_type

    def __call__(self, recon_x, x, mu, logvar):
        # Calculate reconstruction loss
        reconstruction_loss = calc_reconstruction_loss(
            x, recon_x, loss_type=self.loss_type
        )

        # Calculate KL divergence
        kl_divergence = calc_kl(logvar, mu)

        # Calculate total loss
        total_loss = reconstruction_loss + kl_divergence

        return total_loss, reconstruction_loss, kl_divergence


class RA_Loss_Adv:
    def __init__(self, loss_type="bce", discriminator=None, adv_weight=1.0):
        self.loss_type = loss_type
        self.discriminator = discriminator
        self.adv_weight = adv_weight

    def __call__(self, recon_x, x, mu, logvar):
        # Calculate reconstruction loss
        reconstruction_loss = calc_reconstruction_loss(
            x, recon_x, loss_type=self.loss_type
        )

        # Calculate KL divergence
        kl_divergence = calc_kl(logvar, mu)

        # Calculate total loss
        total_loss = reconstruction_loss + kl_divergence

        # Calculate adversarial loss if discriminator is provided
        adversarial_loss = None
        if self.discriminator is not None:
            adversarial_loss = calc_adversarial_loss(recon_x, self.discriminator)
            total_loss += adversarial_loss

        return total_loss, reconstruction_loss, kl_divergence, adversarial_loss


### Helper functions for VAE loss calculation


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce="sum"):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (
        1
        + logvar
        - logvar_o
        - logvar.exp() / torch.exp(logvar_o)
        - (mu - mu_o).pow(2) / torch.exp(logvar_o)
    ).sum(1)
    if reduce == "sum":
        kl = torch.sum(kl)
    elif reduce == "mean":
        kl = torch.mean(kl)
    return kl


def calc_reconstruction_loss(x, recon_x, loss_type="mse", reduction="sum"):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ["sum", "mean", "none"]:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == "mse":
        recon_error = F.mse_loss(recon_x, x, reduction="none")
        recon_error = recon_error.sum(1)
        if reduction == "sum":
            recon_error = recon_error.sum()
        elif reduction == "mean":
            recon_error = recon_error.mean()
    elif loss_type == "l1":
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == "bce":
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_adversarial_loss(recon_x, discriminator):
    # Pass reconstructed images through the discriminator
    # Put the model in evaluation mode
    discriminator.eval()

    # Now you can use the model to make predictions
    with torch.no_grad():
        discriminator_output = discriminator(recon_x)

    # Calculate adversarial loss using binary cross-entropy loss
    adversarial_loss = torch.nn.CrossEntropyLoss()(
        discriminator_output, torch.zeros_like(discriminator_output)
    )

    return adversarial_loss
