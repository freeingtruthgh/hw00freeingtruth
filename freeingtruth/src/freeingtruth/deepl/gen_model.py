import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# For Encoder and Discriminator
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)

# For Generator and Decoder
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder: 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        self.flatten = nn.Flatten()

        self.mu_layer = nn.Linear(512 * 4 * 4, latent_dim)
        self.logvar_layer = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(z.size(0), 512, 4, 4)
        h = self.decoder(h)
        recon = self.output_layer(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

class VAEExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)

# GAN
class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.net = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 512, 4, 4)
        x = self.net(x)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Diffusion
def sinusoidal_embedding(n_steps: int, dim: int) -> torch.Tensor:
    embedding = torch.zeros(n_steps, dim)
    position = torch.arange(n_steps).float().unsqueeze(1)

    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))

    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)

    return embedding


class TimeMLP(nn.Module):
    def __init__(self, time_emb_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


class DiffusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.match_channels = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.residual:
            return out + self.match_channels(x)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim, out_channels)
        self.conv = DiffusionBlock(in_channels, out_channels, residual=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        time_term = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + time_term
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim, out_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DiffusionBlock(out_channels + skip_channels, out_channels, residual=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        time_term = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + time_term
        return x


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 256, n_steps: int = 1000,):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim).to(self.time_embed.weight.device)
        self.time_embed.requires_grad_(False)

        self.input_block = DiffusionBlock(in_channels, base_channels, residual=False)

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)      # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)  # 256 -> 512
        self.down4 = DownBlock(base_channels * 8, base_channels * 8, time_emb_dim)  # 512 -> 512

        self.bottleneck = DiffusionBlock(base_channels * 8, base_channels * 8, residual=True)

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 8, time_emb_dim)
        self.up2 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_emb_dim)
        self.up3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)
        self.up4 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)

        self.output_block = nn.Sequential(
            DiffusionBlock(base_channels, base_channels, residual=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(-1)

        t = t.long()
        t = self.time_embed(t)

        x = self.input_block(x)

        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x, skip3 = self.down3(x, t)
        x, skip4 = self.down4(x, t)

        x = self.bottleneck(x)

        x = self.up1(x, skip4, t)
        x = self.up2(x, skip3, t)
        x = self.up3(x, skip2, t)
        x = self.up4(x, skip1, t)

        x = self.output_block(x)
        return x


class DDPM(nn.Module):
    def __init__(self, network=None, n_steps: int = 1000, min_beta: float = 1e-4, max_beta: float = 0.02, device: torch.device | None = None, image_chw: tuple[int, int, int] = (3, 64, 64)):
        super().__init__()

        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.image_chw = image_chw

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Network selection
        if network is None:
            self.network = DiffusionUNet(in_channels=self.image_chw[0], n_steps = self.n_steps)
        else:
            self.network = network 

        self.network = self.network.to(self.device)

        self.betas = torch.linspace(self.min_beta, self.max_beta, self.n_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, eta: torch.Tensor | None = None) -> torch.Tensor:
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        if eta is None:
            eta = torch.randn(n, c, h, w, device=self.device)

        noisy = (a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eta)
        return noisy

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(x, t)

class DiffusionExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        return self.model.predict_noise(x, t)

class GenModelTrainer(nn.Module):
    def __init__(self, train_loader, val_loader, eta, epochs, loss=None, optimizer=None, optimizer2=None, model_type=None, model=None, model2=None, device=None, latent_dim=128, x=5):
        super().__init__()
        # Training data and hyperparameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta = eta
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.x = x      # How often to save ONNX file

         # Device setup to GPU else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Model type
        if model_type is None:
            self.model_type = "VAE"
        else:
            self.model_type = model_type.upper()    # VAE, GAN, or Diffusion

        if model is None:    
            self.model = VAE(self.latent_dim)       # VAE, Discriminator, or Diffusion
        else:
            self.model = model

        # Loss function
        if loss is None:
            self.loss = VAELoss()
        else:
            self.loss = loss
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        else:
            self.optimizer = optimizer

        # Added model, loss, and optimizer for GAN
        if self.model_type == "GAN":
            if model is None or model2 is None:
                raise ValueError("For GAN, both model and model2 must be provided.")

            #  model2 is for the generator
            self.model2 = model2

            self.discriminator = self.model
            self.generator = self.model2
            
            if optimizer2 is None:
                self.optimizer2 = optim.Adam(self.model2.parameters(), lr=self.eta, betas = (0.5, 0.999))
            else:
                self.optimizer2 = optimizer2
            self.G_loss_vector = torch.zeros(self.epochs)
            self.D_loss_vector = torch.zeros(self.epochs)
            self.G_val_loss_vector = torch.zeros(self.epochs)
            self.D_val_loss_vector = torch.zeros(self.epochs)
        elif self.model_type == "DIFFUSION":
            if model is None:
                raise ValueError("For Diffusion, model must be provided.")

        # set after training, initialize as empty torch Tensors
        self.loss_vector = torch.zeros(self.epochs)
        self.val_loss_vector = torch.zeros(self.epochs)

    def compute_main_loss(self, batch):
        if self.model_type == "VAE":
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(self.device)

            recon_x, mu, logvar = self.model(x)
            loss = self.loss(recon_x, x, mu, logvar)

            if isinstance(loss, tuple):
                loss = loss[0]
            else:
                loss = loss

            return loss

        elif self.model_type == "DIFFUSION":
            x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x0 = x0.to(self.device)

            n = len(x0)
            eta = torch.randn_like(x0, device=self.device)
            t = torch.randint(0, self.model.n_steps, (n,), device=self.device)

            noisy_imgs = self.model(x0, t, eta)
            eta_theta = self.model.predict_noise(noisy_imgs, t)

            loss = self.loss(eta_theta, eta)
            return loss

        elif self.model_type == "GAN":
            real_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            real_imgs = real_imgs.to(self.device)

            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            real_output = self.discriminator(real_imgs)
            d_loss_real = self.loss(real_output, real_labels)

            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_imgs = self.generator(z)

            fake_output = self.discriminator(fake_imgs.detach())
            d_loss_fake = self.loss(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2.0
            return d_loss

        else:
            raise ValueError("Unsupported model_type")

    def compute_model2_loss(self, batch):
        if self.model_type != "GAN":
            raise ValueError("compute_model2_loss is only used for GAN")

        real_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        real_imgs = real_imgs.to(self.device)

        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z)

        output = self.discriminator(fake_imgs)
        g_loss = self.loss(output, real_labels)

        return g_loss

    def validate(self):

        self.model.eval()
        if self.model_type == "GAN":
            self.model2.eval()

        epoch_loss = 0.0
        epoch_loss2 = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):

                if self.model_type == "VAE":
                    x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x = x.to(self.device)

                    recon_x, mu, logvar = self.model(x)
                    loss = self.loss(recon_x, x, mu, logvar)

                    if isinstance(loss, tuple):
                        loss = loss[0]
                    else:
                        loss = loss
                    epoch_loss += loss.item()

                elif self.model_type == "DIFFUSION":
                    x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x0 = x0.to(self.device)

                    n = len(x0)
                    eta = torch.randn_like(x0, device=self.device)
                    t = torch.randint(0, self.model.n_steps, (n,), device=self.device)

                    noisy_imgs = self.model(x0, t, eta)
                    eta_theta = self.model.predict_noise(noisy_imgs, t)

                    loss = self.loss(eta_theta, eta)
                    epoch_loss += loss.item()

                elif self.model_type == "GAN":
                    real_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    real_imgs = real_imgs.to(self.device)

                    batch_size = real_imgs.size(0)
                    real_labels = torch.ones(batch_size, 1, device=self.device)
                    fake_labels = torch.zeros(batch_size, 1, device=self.device)

                    # Discriminator validation loss
                    real_output = self.discriminator(real_imgs)
                    d_loss_real = self.loss(real_output, real_labels)

                    z = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_imgs = self.generator(z)

                    fake_output = self.discriminator(fake_imgs)
                    d_loss_fake = self.loss(fake_output, fake_labels)

                    d_loss = (d_loss_real + d_loss_fake) / 2.0
                    epoch_loss += d_loss.item()

                    # Generator validation loss
                    z = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_imgs = self.generator(z)
                    output = self.discriminator(fake_imgs)

                    g_loss = self.loss(output, real_labels)
                    epoch_loss2 += g_loss.item()

        avg_loss = epoch_loss / len(self.val_loader)

        if self.model_type == "GAN":
            avg_loss2 = epoch_loss2 / len(self.val_loader)
            return avg_loss, avg_loss2

        return avg_loss

    def train(self):
        for ep in range(self.epochs):
            # Training phase
            self.model.train()
            if self.model_type == "GAN":
                self.model2.train()

            epoch_loss = 0.0
            epoch_loss2 = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                loss = self.compute_main_loss(batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if self.model_type == "GAN":
                    
                    self.optimizer2.zero_grad()
                    loss2 = self.compute_model2_loss(batch)
                    loss2.backward()
                    self.optimizer2.step()

                    epoch_loss2 += loss2.item()

                    if batch_idx % 10 == 0:
                        print(
                            f"Epoch {ep+1} | Batch [{batch_idx+1}] "
                            f"D_loss: {loss.item():.4f} | G_loss: {loss2.item():.4f}",
                            flush=True,
                        )
                else:
                    if batch_idx % 10 == 0:
                        print(
                            f"Epoch {ep+1} | Batch [{batch_idx+1}] "
                            f"Loss: {loss.item():.4f}",
                            flush=True,
                        )

            avg_loss = epoch_loss / len(self.train_loader)
            self.loss_vector[ep] = avg_loss

            if self.model_type == "GAN":
                avg_loss2 = epoch_loss2 / len(self.train_loader)
                self.D_loss_vector[ep] = avg_loss
                self.G_loss_vector[ep] = avg_loss2

            # Validation 
            val_result = self.validate()

            if self.model_type == "GAN":
                d_val_loss, g_val_loss = val_result
                self.D_val_loss_vector[ep] = d_val_loss
                self.G_val_loss_vector[ep] = g_val_loss

                print(
                    f"Epoch {ep+1}: "
                    f"Train D_loss={self.D_loss_vector[ep].item():.4f}, "
                    f"Train G_loss={self.G_loss_vector[ep].item():.4f}, "
                    f"Val D_loss={d_val_loss:.4f}, "
                    f"Val G_loss={g_val_loss:.4f}",
                    flush=True,
                )
            else:
                self.val_loss_vector[ep] = val_result

                print(
                    f"Epoch {ep+1}: "
                    f"Train Loss={self.loss_vector[ep].item():.4f}, "
                    f"Val Loss={val_result:.4f}",
                    flush=True,
                )

            # Save ONNX checkpoint every x epochs
            tag = self.model_type.lower()
            if (ep + 1) % self.x == 0:
                self.save_onnx(f"checkpoints/{tag}_checkpoint_epoch_{ep+1}.onnx")
                print(f"Saved ONNX checkpoint at epoch {ep+1}", flush=True)

        self.save_onnx(f"{tag}_final_model.onnx")
        print("Saved final ONNX model.", flush=True)         
        
        if self.model_type == "GAN":
            return self.G_loss_vector, self.D_loss_vector
            
        return self.loss_vector

    def save_onnx(self, filename="model.onnx"):
        import torch

        # Select correct model
        if self.model_type == "GAN":
            model = self.model2  # Generator
        else:
            model = self.model

        model.eval()

        # Move to CPU for export
        model_cpu = model.to("cpu")

        with torch.no_grad():
            if self.model_type == "VAE":
                z = torch.randn(1, self.latent_dim)

                decoder_wrapper = VAEExportWrapper(model_cpu)

                decoder_filename = filename.replace(".onnx", "_decoder.onnx")

                torch.onnx.export(
                    decoder_wrapper,
                    z,
                    decoder_filename,
                    input_names=["latent_vector"],
                    output_names=["generated_image"],
                    dynamic_axes={
                        "latent_vector": {0: "batch_size"},
                        "generated_image": {0: "batch_size"},
                    },
                    opset_version=18,
                )

            elif self.model_type == "DIFFUSION":
                c, h, w = self.model.image_chw
                x = torch.randn(1, c, h, w)
                t = torch.randint(0, self.model.n_steps, (1,), dtype=torch.long)

                wrapper = DiffusionExportWrapper(model_cpu)

                torch.onnx.export(
                    wrapper,
                    (x, t),
                    filename,
                    input_names=["noisy_image", "timestep"],
                    output_names=["predicted_noise"],
                    dynamic_axes={
                        "noisy_image": {0: "batch_size"},
                        "timestep": {0: "batch_size"},
                        "predicted_noise": {0: "batch_size"},
                    },
                    opset_version=18,
                )

            elif self.model_type == "GAN":
                z = torch.randn(1, self.latent_dim)

                torch.onnx.export(
                    model_cpu,
                    z,
                    filename,
                    input_names=["latent_vector"],
                    output_names=["generated_image"],
                    dynamic_axes={
                        "latent_vector": {0: "batch_size"},
                        "generated_image": {0: "batch_size"},
                    },
                    opset_version=18,
                )

            else:
                raise ValueError("Unsupported model_type")

        model.to(self.device)

        print(f"Model exported to {filename}")

