import torch

from models.MambaFNO.v2.fno import FourierLayer2D, FourierNeuralOperator2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_FourierNeuralOperator2D():
    fno = FourierNeuralOperator2D(
        channels_in=3,
        channels_out=5,
        reserved_ratio=0.5,
        weights_size=(16, 32),
        scale_factor=0.5,
    )
    x = torch.randn(10, 3, 32, 64)
    y = fno(x)
    assert y.shape == (10, 5, 16, 32)
    print("FourierNeuralOperator2D test passed.")


def test_FourierLayer2D():
    x = torch.randn(10, 20, 32, 64)
    layer = FourierLayer2D(
        channels_in=x.shape[1],
        channels_out=x.shape[1],
        weights_size=(16, 32),
        scale_factor=0.5,
    )
    y = layer(x)
    assert y.shape == (10, 20, 16, 32)
    print("FourierLayer2D test passed.")


def test_MambaHSIFNO_AvgPool():
    from models.MambaFNO.v2.model import MambaFNO_AvgPool

    x = torch.randn(10, 200, 32, 64).to(device)
    model = MambaFNO_AvgPool(
        in_channels=x.shape[1],
        num_classes=10,
        embedding_channels=128,
        channels_boundary_ratio=[0.3, 0.3, 0.2, 0.2],
        reserved_ratio=1.0,
        weights_size=(128, 64),
    ).to(device)
    y = model(x)
    print(y.shape)
    print("MambaHSIFNO_AvgPool test passed.")


if __name__ == "__main__":
    test_FourierNeuralOperator2D()
    test_FourierLayer2D()
    test_MambaHSIFNO_AvgPool()
