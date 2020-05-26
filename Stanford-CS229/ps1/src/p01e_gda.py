import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m, n = x.shape

        phi = (y == 1).sum()/m
        mu_0 = x[y == 0].sum(axis=0) / (y == 0).sum()
        mu_1 = x[y == 1].sum(axis=0)/(y == 1).sum()
        diff = x.copy()
        diff[y == 0] -= mu_0
        diff[y == 1] -= mu_1
        sigma = (1 / m) * diff.T.dot(diff)
        # Write theta in terms of the parameters
        sigma_inverse = np.linalg.inv(sigma)
        theta = sigma_inverse.dot(mu_1-mu_0)
        theta0 = 0.5 * (mu_0.T.dot(sigma_inverse).dot(mu_0) - mu_1.T.dot(sigma_inverse).dot(mu_1)) - np.log((1 - phi) / phi)
        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta
        return theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def sigmoid(z): return 1 / (1 + np.exp(-z))
        x = util.add_intercept(x)
        probability = sigmoid(x.dot(self.theta))
        predictions = (probability >= 0.5).astype(np.int)
        return predictions
        # *** END CODE HERE
