# TODO: add source links!

import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 (classic structure): C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output

    Matches common descriptions (incl. your Medium link):
      - Input: 32x32 (originally grayscale; for RGB set in_channels=3)
      - C1: 6 feature maps, 5x5 conv, stride 1, no padding (valid)
      - S2: 2x2 average pooling, stride 2
      - C3: 16 feature maps, 5x5 conv, stride 1, no padding
      - S4: 2x2 average pooling, stride 2
      - C5: 120 feature maps, 5x5 conv, stride 1, no padding
            (for 32x32 input, S4 output is 5x5, so C5 becomes 1x1)
      - F6: 84 units
      - Output: num_classes units (originally 10 digits)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 32,
        activation: str = "tanh",
        adapt_to_lenet_geometry: bool = False,
    ) -> None:
        """
        :param in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        :param num_classes: Number of output classes (43 for GTSRB, ...).
        :param input_size: Expected input height/width. Classic LeNet-5 expects 32.
        :param activation: "tanh" (classic) or "relu" (modern variant).
        """
        super().__init__()

        if activation.lower() == "tanh":
            self.act = nn.Tanh()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        # -------------------------
        # C1: 5x5 conv, 6 maps
        # For 32x32 input and padding=0:
        #   output spatial size = 32 - 5 + 1 = 28  (valid convolution)
        # -------------------------
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,          # classic LeNet number of feature maps
            kernel_size=5,           # 5x5 receptive field as in classic LeNet
            stride=1,                # stride 1 keeps maximum spatial detail
            padding=0,               # no padding => "valid" convolution
            bias=True
        )

        # -------------------------
        # S2: 2x2 average pooling, stride 2
        # 28x28 -> 14x14
        # Classic LeNet uses subsampling (average pooling-like).
        # -------------------------
        self.s2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # -------------------------
        # C3: 5x5 conv, 16 maps
        # 14x14 -> 10x10 (valid: 14 - 5 + 1 = 10)
        # Original LeNet had partial connectivity; modern implementations usually use full connectivity.
        # -------------------------
        self.c3 = nn.Conv2d(
            in_channels=6,
            out_channels=16,         # classic LeNet number of maps
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        # -------------------------
        # S4: 2x2 average pooling, stride 2
        # 10x10 -> 5x5
        # -------------------------
        self.s4 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Optional adapter:
        # If input is not 32x32 (e.g. 64x64), after S4 you will NOT get 5x5.
        # This layer forces the geometry back to 5x5 so that C5 can be 5x5 -> 1x1.
        self.adapt = nn.AdaptiveAvgPool2d(
            output_size=(5, 5)
        ) if adapt_to_lenet_geometry else None

        # -------------------------
        # C5: 5x5 conv, 120 maps
        # For classic 32x32 input:
        #   S4 output is 5x5, so 5x5 conv => 1x1.
        # This is why C5 is often described as "convolutional but fully connected".
        # -------------------------
        self.c5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        # -------------------------
        # F6: 84 units (fully connected)
        # After C5 (for classic geometry) tensor is (batch, 120, 1, 1) => flatten => 120 features.
        # -------------------------
        self.f6 = nn.Linear(
            in_features=120,
            out_features=84,
            bias=True
        )

        # -------------------------
        # Output layer
        # Use raw logits (NO softmax here) if you train with nn.CrossEntropyLoss().
        # -------------------------
        self.out = nn.Linear(
            in_features=84,
            out_features=num_classes,
            bias=True
        )

        # Basic sanity check for strict classic geometry
        # (only enforce if user did NOT enable adaptive geometry)
        if (not adapt_to_lenet_geometry) and (input_size != 32):
            raise ValueError(
                f"Classic LeNet-5 expects 32x32 inputs, but got input_size={input_size}. "
                "Either resize your data/transforms to 32x32, or set adapt_to_lenet_geometry=True."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C1 -> activation -> S2
        x = self.c1(x)
        x = self.act(x)
        x = self.s2(x)

        # C3 -> activation -> S4
        x = self.c3(x)
        x = self.act(x)
        x = self.s4(x)

        # Optional: force geometry to 5x5 so C5 becomes 1x1
        if self.adapt is not None:
            x = self.adapt(x)

        # C5 -> activation
        x = self.c5(x)
        x = self.act(x)

        # Flatten (batch, 120, 1, 1) -> (batch, 120)
        x = torch.flatten(x, start_dim=1)

        # F6 -> activation -> Output logits
        x = self.f6(x)
        x = self.act(x)
        x = self.out(x)

        return x
