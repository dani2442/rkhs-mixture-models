"""Exact Gaussian expectations, checked independently of the cumulant formula."""
import itertools
import math
import unittest

import numpy as np
import torch

from src.kernel.polynomial import PolynomialKernel


def normal_moments(mean, variance, order):
    values = [1.0, mean]
    for r in range(2, order + 1):
        values.append(mean * values[-1] + (r - 1) * variance * values[-2])
    return np.array(values[:order + 1])


def gaussian_quadrature(mean, covariance, order):
    """Tensor Gauss--Hermite rule; exact for the polynomials tested below."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    indices = np.array(list(itertools.product(range(order), repeat=len(mean))))
    points = np.sqrt(2) * nodes[indices] @ np.linalg.cholesky(covariance).T + mean
    probabilities = np.prod(weights[indices] / np.sqrt(np.pi), axis=1)
    return points, probabilities


class PolynomialKernelTests(unittest.TestCase):
    def setUp(self):
        self.means = torch.tensor([[0.3, -0.5], [-0.2, 0.7]], dtype=torch.float64)
        self.covariances = torch.tensor(
            [[[0.7, 0.2], [0.2, 0.4]], [[0.3, -0.1], [-0.1, 0.6]]],
            dtype=torch.float64,
        )

    def test_standard_normal_product_moments_through_order_ten(self):
        zero = torch.zeros(1, dtype=torch.float64)
        unit = torch.ones((1, 1), dtype=torch.float64)
        actual = PolynomialKernel()._compute_product_moments(zero, unit, zero, unit, 10)
        expected = [0 if r % 2 else math.prod(range(1, r, 2)) ** 2 for r in range(11)]
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
        self.assertEqual(actual[4].item(), 9.0)
        self.assertEqual(actual[6].item(), 225.0)

    def test_noncentral_scalar_products(self):
        expected = normal_moments(0.4, 0.7, 10) * normal_moments(-0.8, 1.2, 10)
        actual = PolynomialKernel()._compute_product_moments(
            torch.tensor([0.4], dtype=torch.float64),
            torch.tensor([[0.7]], dtype=torch.float64),
            torch.tensor([-0.8], dtype=torch.float64),
            torch.tensor([[1.2]], dtype=torch.float64), 10,
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_component_expectations_against_quadrature(self):
        rules = [gaussian_quadrature(m.numpy(), k.numpy(), 9)
                 for m, k in zip(self.means, self.covariances)]
        for degree in range(1, 9):
            for c in (0.0, 0.6):
                with self.subTest(degree=degree, c=c):
                    actual = PolynomialKernel(degree, c).compute_I(self.means, self.covariances)
                    for k, (x, wx) in enumerate(rules):
                        for s, (y, wy) in enumerate(rules):
                            expected = wx @ ((x @ y.T + c) ** degree) @ wy
                            np.testing.assert_allclose(actual[k, s].item(), expected,
                                                       rtol=2e-12, atol=2e-12)

    def test_data_expectations_against_quadrature(self):
        data = torch.tensor([[0.4, -0.7], [1.1, 0.2]], dtype=torch.float64)
        for degree in (1, 2, 4, 6, 8):
            kernel = PolynomialKernel(degree, c=0.6)
            actual = kernel.compute_J(data, self.means, self.covariances)
            for k, (m, covariance) in enumerate(zip(self.means, self.covariances)):
                points, weights = gaussian_quadrature(m.numpy(), covariance.numpy(), 9)
                expected = ((data.numpy() @ points.T + kernel.c) ** degree) @ weights
                np.testing.assert_allclose(actual[:, k].numpy(), expected,
                                           rtol=2e-12, atol=2e-12)

    def test_singular_covariances(self):
        # Independent noise in orthogonal directions makes the dot product Gaussian.
        m1, m2 = self.means
        k1 = torch.diag(torch.tensor([0.7, 0.0], dtype=torch.float64))
        k2 = torch.diag(torch.tensor([0.0, 0.6], dtype=torch.float64))
        expected = normal_moments((m1 @ m2).item(),
                                  (m1 @ k2 @ m1 + m2 @ k1 @ m2).item(), 8)
        actual = PolynomialKernel()._compute_product_moments(m1, k1, m2, k2, 8)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
        zeros = torch.zeros_like(k1)
        actual = PolynomialKernel()._compute_product_moments(m1, zeros, m2, zeros, 8)
        torch.testing.assert_close(actual, (m1 @ m2) ** torch.arange(9))

    def test_expectation_gradients(self):
        means = self.means.clone().requires_grad_()
        factors = torch.linalg.cholesky(self.covariances).requires_grad_()
        data = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
        for degree in (2, 4, 6):
            kernel = PolynomialKernel(degree, c=0.6)

            def expectations(m, factors):
                covariance = factors @ factors.transpose(-1, -2)
                return kernel.compute_I(m, covariance), kernel.compute_J(data, m, covariance)

            with self.subTest(degree=degree):
                self.assertTrue(torch.autograd.gradcheck(expectations, (means, factors)))

    def test_degree_must_be_positive_integer(self):
        for degree in (0, -1, 1.5, True):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                PolynomialKernel(degree)


if __name__ == "__main__":
    unittest.main()
