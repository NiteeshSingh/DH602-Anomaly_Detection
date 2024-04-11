import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        input_shape (tuple): Shape of the input images (channels, height, width).
        hidden_dims (list): List of integers representing the dimensions of hidden layers.
        latent_dim (int): Dimension of the latent space.
        output_shape (tuple): Shape of the output images (channels, height, width).

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        latent_dim (int): Dimension of the latent space.
        height (int): Height of the feature maps.
        width (int): Width of the feature maps.
        fc_mu (nn.Linear): Linear layer for calculating mean of latent space.
        fc_logvar (nn.Linear): Linear layer for calculating log variance of latent space.
    """

    def __init__(self, input_shape, hidden_dims, latent_dim, output_shape):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.spatial_dims = [
            (input_shape[1], input_shape[2])
        ]  # [(height, width)]-> at each encoder level
        self.encoder = self._build_encoder(input_shape, hidden_dims, latent_dim)
        self.fc_mu = nn.Linear(
            hidden_dims[-1] * self.spatial_dims[-1][0] * self.spatial_dims[-1][1],
            latent_dim,
        )  # mean of latent space
        self.fc_logvar = nn.Linear(
            hidden_dims[-1] * self.spatial_dims[-1][0] * self.spatial_dims[-1][1],
            latent_dim,
        )  # log variance of latent space
        self.decoder = self._build_decoder(latent_dim, hidden_dims[::-1], output_shape)

    def _build_encoder(self, input_shape, hidden_dims, latent_dim):
        """
        Build the encoder module.

        Returns:
            Encoder: Encoder module.
        """
        encoder = nn.Sequential()
        in_dim = input_shape[0]
        height, width = input_shape[1], input_shape[2]
        for idx, out_channels in enumerate(hidden_dims):
            encoder.add_module(
                f"conv{idx+1}",
                nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=2, padding=1),
            )
            encoder.add_module(f"bn{idx+1}", nn.BatchNorm2d(out_channels))
            encoder.add_module(f"relu{idx+1}", nn.ReLU())

            in_dim = out_channels
            height, width = self.calculate_new_size(height, width)
            self.spatial_dims.append((height, width))
        encoder.add_module("flatten", nn.Flatten())
        return encoder

    def _build_decoder(self, latent_dim, hidden_dims, output_shape):
        """
        Build the decoder module.

        Returns:
            Decoder: Decoder module.
        """
        decoder = nn.Sequential()
        in_dim = hidden_dims[0]
        height, width = self.spatial_dims[-1][0], self.spatial_dims[-1][1]
        decoder.add_module("fc", nn.Linear(latent_dim, hidden_dims[0] * height * width))
        decoder.add_module("relu", nn.ReLU())
        decoder.add_module("reshape", nn.Unflatten(1, (hidden_dims[0], height, width)))
        for idx, out_channels in enumerate(hidden_dims):
            new_height, new_width = (
                self.spatial_dims[-2 - idx][0],
                self.spatial_dims[-2 - idx][1],
            )
            if idx == len(hidden_dims) - 1:
                decoder.add_module(
                    f"conv_transpose_output",
                    nn.ConvTranspose2d(
                        in_dim,
                        output_shape[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=(
                            1 - (2 * height - new_height),
                            1 - (2 * width - new_width),
                        ),
                    ),
                )
                decoder.add_module("sigmoid", nn.Sigmoid())
            else:
                decoder.add_module(
                    f"conv_transpose{idx+1}",
                    nn.ConvTranspose2d(
                        in_dim,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=(
                            1 - (2 * height - new_height),
                            1 - (2 * width - new_width),
                        ),
                    ),
                )
                decoder.add_module(f"bn{idx+1}", nn.BatchNorm2d(out_channels))
                decoder.add_module(f"relu{idx+1}", nn.ReLU())
                in_dim = out_channels
                height, width = new_height, new_width
        return decoder

    def calculate_new_size(self, height, width):
        """
        Calculate the new size of the feature maps after convolution.

        Returns:
            int: New height.
            int: New width.
        """
        return (height + 2 - 3) // 2 + 1, (width + 2 - 3) // 2 + 1

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Returns:
            torch.Tensor: Reconstructed output.
            torch.Tensor: Mean of latent space.
            torch.Tensor: Log variance of latent space.
        """
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class VAE_Attention(nn.Module):
    """
    Variational Autoencoder (VAE) model with attention mechanism.

    Args:
        input_shape (tuple): Shape of the input images (channels, height, width).
        hidden_dims (list): List of integers representing the dimensions of hidden layers.
        latent_dim (int): Dimension of the latent space.
        output_shape (tuple): Shape of the output images (channels, height, width).

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        latent_dim (int): Dimension of the latent space.
        height (int): Height of the feature maps.
        width (int): Width of the feature maps.
        fc_mu (nn.Linear): Linear layer for calculating mean of latent space.
        fc_logvar (nn.Linear): Linear layer for calculating log variance of latent space.
    """

    def __init__(self, input_shape, hidden_dims, latent_dim, output_shape):
        super(VAE_Attention, self).__init__()
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.spatial_dims = [
            (input_shape[1], input_shape[2])
        ]  # [(height, width)]-> at each encoder level
        self.encoder = self._build_encoder(input_shape, hidden_dims, latent_dim)
        self.fc_mu = nn.Linear(
            hidden_dims[-1] * self.spatial_dims[-1][0] * self.spatial_dims[-1][1],
            latent_dim,
        )  # mean of latent space
        self.fc_logvar = nn.Linear(
            hidden_dims[-1] * self.spatial_dims[-1][0] * self.spatial_dims[-1][1],
            latent_dim,
        )  # log variance of latent space
        self.decoder = self._build_decoder(latent_dim, hidden_dims[::-1], output_shape)

    def _build_encoder(self, input_shape, hidden_dims, latent_dim):
        """
        Build the encoder module with attention mechanism.

        Returns:
            Encoder: Encoder module.
        """
        encoder = nn.Sequential()
        in_dim = input_shape[0]
        height, width = input_shape[1], input_shape[2]
        for idx, out_channels in enumerate(hidden_dims):
            encoder.add_module(
                f"conv{idx + 1}",
                nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=2, padding=1),
            )
            encoder.add_module(f"bn{idx + 1}", nn.BatchNorm2d(out_channels))
            encoder.add_module(f"relu{idx + 1}", nn.ReLU())

            # Attention mechanism
            encoder.add_module(f"attention{idx + 1}", SelfAttention(out_channels))

            in_dim = out_channels
            height, width = self.calculate_new_size(height, width)
            self.spatial_dims.append((height, width))
        encoder.add_module("flatten", nn.Flatten())
        return encoder

    def _build_decoder(self, latent_dim, hidden_dims, output_shape):
        """
        Build the decoder module with attention mechanism.

        Returns:
            Decoder: Decoder module.
        """
        decoder = nn.Sequential()
        in_dim = hidden_dims[0]
        height, width = self.spatial_dims[-1][0], self.spatial_dims[-1][1]
        decoder.add_module("fc", nn.Linear(latent_dim, hidden_dims[0] * height * width))
        decoder.add_module("relu", nn.ReLU())
        decoder.add_module("reshape", nn.Unflatten(1, (hidden_dims[0], height, width)))
        for idx, out_channels in enumerate(hidden_dims):
            new_height, new_width = (
                self.spatial_dims[-2 - idx][0],
                self.spatial_dims[-2 - idx][1],
            )

            if idx == len(hidden_dims) - 1:
                decoder.add_module(
                    f"conv_transpose_output",
                    nn.ConvTranspose2d(
                        in_dim,
                        output_shape[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=(
                            1 - (2 * height - new_height),
                            1 - (2 * width - new_width),
                        ),
                    ),
                )
                decoder.add_module("sigmoid", nn.Sigmoid())
            else:
                decoder.add_module(
                    f"conv_transpose{idx + 1}",
                    nn.ConvTranspose2d(
                        in_dim,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=(
                            1 - (2 * height - new_height),
                            1 - (2 * width - new_width),
                        ),
                    ),
                )
                decoder.add_module(f"bn{idx + 1}", nn.BatchNorm2d(out_channels))
                decoder.add_module(f"relu{idx + 1}", nn.ReLU())
                # Attention mechanism
                decoder.add_module(f"attention{idx + 1}", SelfAttention(out_channels))
                in_dim = out_channels
                height, width = new_height, new_width
        return decoder

    def calculate_new_size(self, height, width):
        """
        Calculate the new size of the feature maps after convolution.

        Returns:
            int: New height.
            int: New width.
        """
        return (height + 2 - 3) // 2 + 1, (width + 2 - 3) // 2 + 1

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Returns:
            torch.Tensor: Reconstructed output.
            torch.Tensor: Mean of latent space.
            torch.Tensor: Log variance of latent space.
        """
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class SelfAttention(nn.Module):
    """
    Self Attention mechanism.

    Args:
        in_dim (int): Input feature dimension.

    Attributes:
        conv_query (nn.Conv2d): Convolutional layer for query computation.
        conv_key (nn.Conv2d): Convolutional layer for key computation.
        conv_value (nn.Conv2d): Convolutional layer for value computation.
    """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = (
            self.conv_query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        )  # B x (H*W) x C
        key = self.conv_key(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        energy = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = self.softmax(energy)
        value = self.conv_value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(
            batch_size, channels, height, width
        )
        return out
