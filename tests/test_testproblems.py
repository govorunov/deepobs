"""Comprehensive tests for all 26 DeepOBS PyTorch test problems."""

import pytest
import torch
import torch.nn as nn

from deepobs.pytorch import testproblems
from tests.test_utils import set_seed, get_available_test_problems


# Get all available test problems
AVAILABLE_PROBLEMS, MANUAL_PROBLEMS = get_available_test_problems()


class TestProblemBase:
    """Base tests that should work for all test problems."""

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_problem_instantiation(self, problem_name):
        """Test that problem can be instantiated."""
        try:
            problem = testproblems.testproblem(problem_name, batch_size=32)
            assert problem is not None
        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")
        except Exception as e:
            pytest.fail(f"Failed to instantiate {problem_name}: {e}")

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_problem_has_model(self, problem_name):
        """Test that problem has a model."""
        try:
            problem = testproblems.testproblem(problem_name, batch_size=32)
            assert hasattr(problem, 'model')
            assert problem.model is not None
            assert isinstance(problem.model, nn.Module)
        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_problem_has_data_loaders(self, problem_name):
        """Test that problem has train and test data loaders."""
        try:
            problem = testproblems.testproblem(problem_name, batch_size=32)
            assert hasattr(problem, 'train_loader')
            assert hasattr(problem, 'test_loader')
            assert problem.train_loader is not None
            assert problem.test_loader is not None
        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_forward_pass(self, problem_name):
        """Test that forward pass works."""
        try:
            set_seed(42)
            problem = testproblems.testproblem(problem_name, batch_size=32)

            # Get a batch
            batch = next(iter(problem.train_loader))
            x, y = batch

            # Forward pass
            problem.model.train()
            output = problem.model(x)

            # Check output is a tensor
            assert isinstance(output, (torch.Tensor, tuple))

        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_backward_pass(self, problem_name):
        """Test that backward pass works."""
        try:
            set_seed(42)
            problem = testproblems.testproblem(problem_name, batch_size=32)

            # Get a batch
            batch = next(iter(problem.train_loader))
            x, y = batch

            # Forward and backward pass
            problem.model.train()
            losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

            # Compute mean loss
            loss = losses.mean()

            # Add regularization if available
            if hasattr(problem, 'get_regularization_loss'):
                reg_loss = problem.get_regularization_loss()
                if reg_loss is not None:
                    loss = loss + reg_loss

            # Backward pass
            loss.backward()

            # Check gradients exist
            has_gradients = False
            for param in problem.model.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break

            assert has_gradients, f"No gradients found for {problem_name}"

        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")


class TestMNISTProblems:
    """Tests for MNIST-based problems."""

    def test_mnist_logreg(self):
        """Test MNIST logistic regression problem."""
        problem = testproblems.testproblem('mnist_logreg', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)
        assert 0.0 <= accuracy <= 1.0

    def test_mnist_mlp(self):
        """Test MNIST MLP problem."""
        problem = testproblems.testproblem('mnist_mlp', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)
        assert 0.0 <= accuracy <= 1.0

    def test_mnist_2c2d(self):
        """Test MNIST 2C2D problem."""
        problem = testproblems.testproblem('mnist_2c2d', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)

    def test_mnist_vae(self):
        """Test MNIST VAE problem."""
        problem = testproblems.testproblem('mnist_vae', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)
        # VAE doesn't compute accuracy
        assert accuracy is None or accuracy == 0.0


class TestFashionMNISTProblems:
    """Tests for Fashion-MNIST-based problems."""

    def test_fmnist_logreg(self):
        """Test Fashion-MNIST logistic regression problem."""
        problem = testproblems.testproblem('fmnist_logreg', batch_size=128)
        assert problem is not None

    def test_fmnist_mlp(self):
        """Test Fashion-MNIST MLP problem."""
        problem = testproblems.testproblem('fmnist_mlp', batch_size=128)
        assert problem is not None

    def test_fmnist_2c2d(self):
        """Test Fashion-MNIST 2C2D problem."""
        problem = testproblems.testproblem('fmnist_2c2d', batch_size=128)
        assert problem is not None

    def test_fmnist_vae(self):
        """Test Fashion-MNIST VAE problem."""
        problem = testproblems.testproblem('fmnist_vae', batch_size=128)
        assert problem is not None


class TestCIFAR10Problems:
    """Tests for CIFAR-10-based problems."""

    def test_cifar10_3c3d(self):
        """Test CIFAR-10 3C3D problem."""
        problem = testproblems.testproblem('cifar10_3c3d', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)

    def test_cifar10_vgg16(self):
        """Test CIFAR-10 VGG16 problem."""
        problem = testproblems.testproblem('cifar10_vgg16', batch_size=64)
        assert problem is not None

    def test_cifar10_vgg19(self):
        """Test CIFAR-10 VGG19 problem."""
        problem = testproblems.testproblem('cifar10_vgg19', batch_size=64)
        assert problem is not None


class TestCIFAR100Problems:
    """Tests for CIFAR-100-based problems."""

    def test_cifar100_3c3d(self):
        """Test CIFAR-100 3C3D problem."""
        problem = testproblems.testproblem('cifar100_3c3d', batch_size=128)
        assert problem is not None

    def test_cifar100_allcnnc(self):
        """Test CIFAR-100 All-CNN-C problem."""
        problem = testproblems.testproblem('cifar100_allcnnc', batch_size=128)
        assert problem is not None

    def test_cifar100_vgg16(self):
        """Test CIFAR-100 VGG16 problem."""
        problem = testproblems.testproblem('cifar100_vgg16', batch_size=64)
        assert problem is not None

    def test_cifar100_vgg19(self):
        """Test CIFAR-100 VGG19 problem."""
        problem = testproblems.testproblem('cifar100_vgg19', batch_size=64)
        assert problem is not None

    def test_cifar100_wrn404(self):
        """Test CIFAR-100 WRN-40-4 problem."""
        problem = testproblems.testproblem('cifar100_wrn404', batch_size=64)
        assert problem is not None


class TestSVHNProblems:
    """Tests for SVHN-based problems."""

    def test_svhn_3c3d(self):
        """Test SVHN 3C3D problem."""
        problem = testproblems.testproblem('svhn_3c3d', batch_size=128)
        assert problem is not None

    def test_svhn_wrn164(self):
        """Test SVHN WRN-16-4 problem."""
        problem = testproblems.testproblem('svhn_wrn164', batch_size=64)
        assert problem is not None


class TestSyntheticProblems:
    """Tests for synthetic test problems."""

    def test_quadratic_deep(self):
        """Test quadratic deep problem."""
        problem = testproblems.testproblem('quadratic_deep', batch_size=128)
        assert problem is not None

        batch = next(iter(problem.train_loader))
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

        assert losses.shape == (128,)
        # Quadratic problem doesn't have accuracy
        assert accuracy is None or accuracy == 0.0

    def test_two_d_rosenbrock(self):
        """Test 2D Rosenbrock problem."""
        problem = testproblems.testproblem('two_d_rosenbrock', batch_size=128)
        assert problem is not None

    def test_two_d_beale(self):
        """Test 2D Beale problem."""
        problem = testproblems.testproblem('two_d_beale', batch_size=128)
        assert problem is not None

    def test_two_d_branin(self):
        """Test 2D Branin problem."""
        problem = testproblems.testproblem('two_d_branin', batch_size=128)
        assert problem is not None


@pytest.mark.skip(reason="ImageNet requires manual download")
class TestImageNetProblems:
    """Tests for ImageNet-based problems."""

    def test_imagenet_vgg16(self):
        """Test ImageNet VGG16 problem."""
        try:
            problem = testproblems.testproblem('imagenet_vgg16', batch_size=32)
            assert problem is not None
        except FileNotFoundError:
            pytest.skip("ImageNet data not available")

    def test_imagenet_vgg19(self):
        """Test ImageNet VGG19 problem."""
        try:
            problem = testproblems.testproblem('imagenet_vgg19', batch_size=32)
            assert problem is not None
        except FileNotFoundError:
            pytest.skip("ImageNet data not available")

    def test_imagenet_inception_v3(self):
        """Test ImageNet Inception V3 problem."""
        try:
            problem = testproblems.testproblem('imagenet_inception_v3', batch_size=16)
            assert problem is not None
        except FileNotFoundError:
            pytest.skip("ImageNet data not available")


@pytest.mark.skip(reason="Tolstoi requires manual download")
class TestTolstoiProblems:
    """Tests for Tolstoi-based problems."""

    def test_tolstoi_char_rnn(self):
        """Test Tolstoi character RNN problem."""
        try:
            problem = testproblems.testproblem('tolstoi_char_rnn', batch_size=32)
            assert problem is not None
        except FileNotFoundError:
            pytest.skip("Tolstoi data not available")


class TestProblemModes:
    """Test train/eval mode switching for test problems."""

    @pytest.mark.parametrize("problem_name", [
        'mnist_mlp', 'mnist_2c2d', 'fmnist_mlp', 'cifar10_3c3d'
    ])
    def test_train_eval_modes(self, problem_name):
        """Test that problems can switch between train and eval modes."""
        problem = testproblems.testproblem(problem_name, batch_size=32)

        # Train mode
        problem.model.train()
        assert problem.model.training

        # Eval mode
        problem.model.eval()
        assert not problem.model.training


class TestProblemReproducibility:
    """Test reproducibility with fixed seeds."""

    def test_reproducible_initialization(self):
        """Test that model initialization is reproducible."""
        set_seed(42)
        problem1 = testproblems.testproblem('mnist_mlp', batch_size=32)

        set_seed(42)
        problem2 = testproblems.testproblem('mnist_mlp', batch_size=32)

        # Compare first layer weights
        params1 = list(problem1.model.parameters())[0]
        params2 = list(problem2.model.parameters())[0]

        assert torch.allclose(params1, params2)

    def test_reproducible_forward_pass(self):
        """Test that forward pass is reproducible."""
        set_seed(42)
        problem = testproblems.testproblem('mnist_logreg', batch_size=32)

        set_seed(42)
        batch = next(iter(problem.train_loader))
        output1 = problem.model(batch[0])

        set_seed(42)
        batch = next(iter(problem.train_loader))
        output2 = problem.model(batch[0])

        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
