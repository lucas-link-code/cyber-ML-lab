# EXERCISE NAME: 14 | Backpropagation without a framework

**OBJECTIVE:** Observe forward propagation, loss, gradients and weight updates in a tiny NumPy model.

**SCENARIO:** Compare how learning differs in a neural network from fitting a linear classifier.

**DATASET:** A four-row, programmatically generated XOR dataset (no external CSV; toy appendix after cybersecurity core).

**ALGORITHM:** Two-layer feed-forward network: tanh hidden layer and sigmoid output; binary cross-entropy loss.

**COMMAND:** Run from the lab folder with your virtual environment activated.

```bash
python scripts/neural_network_intro.py --epochs 3000 --rate 0.4 --name nn_default
```

**WHAT THE COMMAND DOES:** Computes hidden activations `tanh(XW1+b1)`, a sigmoid output, binary cross-entropy, derivatives and gradient-descent updates using NumPy only.

**EXPECTED OUTPUT (ILLUSTRATIVE, NOT VERIFIED OUTPUT):** Illustrative only: loss falls across epochs and XOR probabilities approach `[0,1,1,0]`. Check `outputs/nn_default_neural_metrics.json`.

**WHAT TO OBSERVE:** Look at initial vs final loss and the shape of the training-loss curve. This is not a production malware model.

**INVESTIGATION:** 1. Why can a network with a hidden nonlinear layer learn XOR? 2. What does the learning rate multiply?

**IMPROVEMENT:** Change **only learning rate**: `python scripts/neural_network_intro.py --epochs 3000 --rate 0.05 --name nn_slow`.

**COMPARISON:** Compare `outputs/nn_default_neural_metrics.json` with `outputs/nn_slow_neural_metrics.json` and the two loss charts.

**LESSON:** Backpropagation uses the chain rule to compute how each weight contributes to loss; learning rate controls step size, not the learned target. Both too-small and too-large rates can harm convergence.
