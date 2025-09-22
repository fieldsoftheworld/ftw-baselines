from torch import Tensor
from torchgeo.models import FCSiamDiff


class FCSiamAvg(FCSiamDiff):
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input images of shape (b, t, c, h, w)

        Returns:
            predicted change masks of size (b, classes, h, w)
        """
        x1, x2 = x[:, 0], x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [(features2[i] + features1[i]) / 2 for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks
