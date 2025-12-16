"""Comprehensive tests for all DeepOBS PyTorch architectures."""

import pytest
import torch
import torch.nn as nn

from deepobs.pytorch.testproblems import (
    _logreg, _mlp, _2c2d, _3c3d, _vgg, _wrn,
    _inception_v3, _vae, _allcnnc, tolstoi_char_rnn
)
from tests.test_utils import (
    set_seed, assert_shape, count_parameters,
    check_gpu_available, get_dummy_batch, get_dummy_sequence_batch
)


class TestLogisticRegression:
    """Tests for Logistic Regression architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        model = _logreg.logreg_model(num_outputs=10)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = _logreg.logreg_model(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)

        output = model(x)
        assert_shape(output, (32, 10), "logreg output")

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = _logreg.logreg_model(num_outputs=10)
        num_params = count_parameters(model)

        # 784 * 10 + 10 = 7850
        expected = 784 * 10 + 10
        assert num_params == expected

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = _logreg.logreg_model(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None


class TestMLP:
    """Tests for Multi-Layer Perceptron architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        model = _mlp.mlp_model(num_outputs=10)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = _mlp.mlp_model(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)

        output = model(x)
        assert_shape(output, (32, 10), "MLP output")

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        model = _mlp.mlp_model(num_outputs=10)

        for batch_size in [1, 16, 32, 64, 128]:
            x = torch.randn(batch_size, 1, 28, 28)
            output = model(x)
            assert_shape(output, (batch_size, 10), f"MLP output (batch={batch_size})")

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = _mlp.mlp_model(num_outputs=10)
        num_params = count_parameters(model)

        # Should be several hundred thousand parameters
        assert 100000 < num_params < 1000000


class Test2C2D:
    """Tests for 2C2D architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        model = _2c2d.two_c_two_d_model(num_outputs=10)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = _2c2d.two_c_two_d_model(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)

        output = model(x)
        assert_shape(output, (32, 10), "2C2D output")

    def test_gradient_flow(self):
        """Test gradients flow through conv and dense layers."""
        model = _2c2d.two_c_two_d_model(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))

        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class Test3C3D:
    """Tests for 3C3D architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        model = _3c3d.three_c_three_d_model(num_outputs=10)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with CIFAR-10 sized input."""
        model = _3c3d.three_c_three_d_model(num_outputs=10)
        x = torch.randn(32, 3, 32, 32)

        output = model(x)
        assert_shape(output, (32, 10), "3C3D output")

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = _3c3d.three_c_three_d_model(num_outputs=10)
        num_params = count_parameters(model)

        # Should be several hundred thousand parameters
        assert 100000 < num_params < 5000000


class TestVGG:
    """Tests for VGG architectures."""

    def test_vgg16_creation(self):
        """Test VGG16 can be created."""
        model = _vgg.vgg16_model(num_outputs=10)
        assert model is not None

    def test_vgg16_forward_pass(self):
        """Test VGG16 forward pass."""
        model = _vgg.vgg16_model(num_outputs=10)
        x = torch.randn(8, 3, 32, 32)  # Smaller batch for VGG

        output = model(x)
        assert_shape(output, (8, 10), "VGG16 output")

    def test_vgg19_creation(self):
        """Test VGG19 can be created."""
        model = _vgg.vgg19_model(num_outputs=10)
        assert model is not None

    def test_vgg19_forward_pass(self):
        """Test VGG19 forward pass."""
        model = _vgg.vgg19_model(num_outputs=10)
        x = torch.randn(8, 3, 32, 32)

        output = model(x)
        assert_shape(output, (8, 10), "VGG19 output")

    def test_vgg16_parameter_count(self):
        """Test VGG16 has more parameters than VGG19."""
        model16 = _vgg.vgg16_model(num_outputs=10)
        model19 = _vgg.vgg19_model(num_outputs=10)

        params16 = count_parameters(model16)
        params19 = count_parameters(model19)

        # VGG19 should have more parameters
        assert params19 > params16


class TestWRN:
    """Tests for Wide Residual Network."""

    def test_wrn_creation(self):
        """Test WRN can be created."""
        # WRN-16-4
        model = _wrn.wrn_model(depth=16, width=4, num_outputs=10)
        assert model is not None

    def test_wrn_forward_pass(self):
        """Test WRN forward pass."""
        model = _wrn.wrn_model(depth=16, width=4, num_outputs=10)
        x = torch.randn(16, 3, 32, 32)

        output = model(x)
        assert_shape(output, (16, 10), "WRN output")

    def test_wrn_train_mode(self):
        """Test WRN in train mode (batch norm)."""
        model = _wrn.wrn_model(depth=16, width=4, num_outputs=10)
        model.train()

        x = torch.randn(16, 3, 32, 32)
        output = model(x)

        assert output is not None

    def test_wrn_eval_mode(self):
        """Test WRN in eval mode (batch norm)."""
        model = _wrn.wrn_model(depth=16, width=4, num_outputs=10)
        model.eval()

        x = torch.randn(16, 3, 32, 32)
        output = model(x)

        assert output is not None


class TestInceptionV3:
    """Tests for Inception V3."""

    def test_inception_creation(self):
        """Test Inception V3 can be created."""
        model = _inception_v3.inception_v3_model(num_outputs=1000)
        assert model is not None

    @pytest.mark.slow
    def test_inception_forward_pass(self):
        """Test Inception V3 forward pass."""
        model = _inception_v3.inception_v3_model(num_outputs=1000)
        x = torch.randn(4, 3, 299, 299)  # Small batch, large input

        output = model(x)
        assert_shape(output, (4, 1000), "Inception V3 output")

    def test_inception_parameter_count(self):
        """Test Inception V3 has many parameters."""
        model = _inception_v3.inception_v3_model(num_outputs=1000)
        num_params = count_parameters(model)

        # Inception V3 has ~20M+ parameters
        assert num_params > 10000000


class TestVAE:
    """Tests for Variational Autoencoder."""

    def test_vae_creation(self):
        """Test VAE can be created."""
        model = _vae.vae_model()
        assert model is not None

    def test_vae_forward_pass(self):
        """Test VAE forward pass."""
        model = _vae.vae_model()
        x = torch.randn(32, 1, 28, 28)

        reconstruction, mean, logvar = model(x)

        assert_shape(reconstruction, (32, 1, 28, 28), "VAE reconstruction")
        assert_shape(mean, (32, 20), "VAE mean")  # latent_dim=20
        assert_shape(logvar, (32, 20), "VAE logvar")

    def test_vae_train_eval_modes(self):
        """Test VAE in train and eval modes."""
        model = _vae.vae_model()
        x = torch.randn(32, 1, 28, 28)

        # Train mode
        model.train()
        recon_train, _, _ = model(x)
        assert recon_train is not None

        # Eval mode
        model.eval()
        recon_eval, _, _ = model(x)
        assert recon_eval is not None


class TestAllCNNC:
    """Tests for All-CNN-C architecture."""

    def test_allcnnc_creation(self):
        """Test All-CNN-C can be created."""
        model = _allcnnc.allcnnc_model(num_outputs=100)
        assert model is not None

    def test_allcnnc_forward_pass(self):
        """Test All-CNN-C forward pass."""
        model = _allcnnc.allcnnc_model(num_outputs=100)
        x = torch.randn(32, 3, 32, 32)

        output = model(x)
        assert_shape(output, (32, 100), "All-CNN-C output")

    def test_allcnnc_no_pooling(self):
        """Test that All-CNN-C uses strided conv instead of pooling."""
        model = _allcnnc.allcnnc_model(num_outputs=100)

        # Check that there are no MaxPool layers
        has_pooling = any(isinstance(m, nn.MaxPool2d) for m in model.modules())
        assert not has_pooling, "All-CNN-C should not have max pooling layers"


class TestCharRNN:
    """Tests for Character-level RNN."""

    def test_char_rnn_creation(self):
        """Test Char RNN can be created."""
        model = tolstoi_char_rnn.char_rnn_model(vocab_size=83)
        assert model is not None

    def test_char_rnn_forward_pass(self):
        """Test Char RNN forward pass."""
        model = tolstoi_char_rnn.char_rnn_model(vocab_size=83)
        x = torch.randint(0, 83, (32, 50))  # batch_size=32, seq_len=50

        output, hidden = model(x)
        assert_shape(output, (32, 50, 83), "Char RNN output")
        assert hidden is not None

    def test_char_rnn_hidden_state(self):
        """Test Char RNN hidden state persistence."""
        model = tolstoi_char_rnn.char_rnn_model(vocab_size=83)
        x = torch.randint(0, 83, (32, 50))

        # First forward pass
        output1, hidden1 = model(x)

        # Second forward pass with hidden state
        output2, hidden2 = model(x, hidden1)

        # Outputs should be different when using hidden state
        assert not torch.allclose(output1, output2)


# Parametrized tests for all architectures
@pytest.mark.parametrize("model_fn,input_shape,output_size", [
    (_logreg.logreg_model, (16, 1, 28, 28), 10),
    (_mlp.mlp_model, (16, 1, 28, 28), 10),
    (_2c2d.two_c_two_d_model, (16, 1, 28, 28), 10),
    (_3c3d.three_c_three_d_model, (16, 3, 32, 32), 10),
    (_allcnnc.allcnnc_model, (16, 3, 32, 32), 100),
])
def test_architecture_backward_pass(model_fn, input_shape, output_size):
    """Test that backward pass works for all architectures."""
    model = model_fn(num_outputs=output_size)
    x = torch.randn(*input_shape)
    y = torch.randint(0, output_size, (input_shape[0],))

    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()

    # Verify all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.skipif(not check_gpu_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_fn,input_shape,output_size", [
    (_logreg.logreg_model, (16, 1, 28, 28), 10),
    (_mlp.mlp_model, (16, 1, 28, 28), 10),
    (_2c2d.two_c_two_d_model, (16, 1, 28, 28), 10),
])
def test_architecture_gpu(model_fn, input_shape, output_size):
    """Test architectures work on GPU."""
    device = torch.device('cuda')
    model = model_fn(num_outputs=output_size).to(device)
    x = torch.randn(*input_shape).to(device)

    output = model(x)
    assert output.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
