import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_train)
    plot = util.plot(x_val, y_val, model.theta, f'{pred_path}.png')
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        def sigmoid(z): return 1/(1+np.exp(-z))

        while True:
            theta = self.theta
            n = x.dot(theta)
            # Computing the derivative of J
            J = - (1 / m) * (y - sigmoid(n)).dot(x)

            # Compute Hessian Inverse
            H_inv = np.linalg.inv(
                (1/m) * sigmoid(n).dot(sigmoid(1-n)) * (x.T).dot(x))

            # Update Model
            self.theta = theta - H_inv.dot(J)

            if(np.linalg.norm(self.theta - theta, ord=1) < self.eps):
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def sigmoid(z): return 1/(1+np.exp(-z))
        return sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***
