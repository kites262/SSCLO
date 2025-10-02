import torch

from model.MambaHSIFNO import FourierLayer2D, FourierNeuralOperator2D, MambaHSIFNO
from utils.data_load_operate import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_FourierNeuralOperator2D():
    fno = FourierNeuralOperator2D(
        in_channels=3,
        out_channels=5,
        in_y=16,
        in_x=32,
        reserved_x=10,
        reserved_y=16,
    )
    x = torch.randn(1, 3, 16, 32)
    y = fno(x)
    assert y.shape == (1, 5, 16, 32)
    print("FourierNeuralOperator2D test passed.")


def test_FourierLayer2D():
    fno_layer = FourierLayer2D(out_channels=12)
    x = torch.randn(2, 10, 32, 64)
    y = fno_layer(x)
    assert y.shape == (2, 12, 32, 64)
    print("FourierLayer2D test passed.")


def test_MambaHSIFNO():
    data, _ = load_data("HanChuan")
    print(f"Data shape: {data.shape}")

    data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    data = data.to(device)
    _, C, _, _ = data.shape

    model = MambaHSIFNO(in_channels=C, embedding_channels=64, num_classes=10).to(device)
    output = model(data)
    print(f"Output shape: {output.shape}")


def test_MambaHSI():
    data, _ = load_data("HanChuan")
    print(f"Data shape: {data.shape}")

    data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    data = data.to(device)
    _, C, _, _ = data.shape

    from model.MambaHSI import MambaHSI

    model = MambaHSI(in_channels=C, hidden_dim=64, num_classes=10).to(device)
    output = model(data)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    # test_FourierNeuralOperator2D()
    # test_FourierLayer2D()
    test_MambaHSIFNO()
    # test_MambaHSI()
