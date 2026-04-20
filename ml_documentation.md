# ML Documentation and Discussion

## Project Goal and Context
The goal is to improve accuracy of two mobile-robot sensors using a neural network. The sensors are mounted 0.5 m apart and each returns time and 3D position (x, y, z). There is measurement noise in the raw outputs. The model is trained on windows of 50 samples so it can detect outliers and learn correction patterns. Training data are normalized by shifting their average (centering) before learning.

## Data and Input Representation
- Each training sample is a window of 50 readings.
- Each reading has 4 values: time, x, y, z.
- Total input size per sample is 50 x 4 = 200 features.
- Two datasets are used in parallel: one for sensor 1 and one for sensor 2.

## Model Architecture
The model is a fully connected MLP with Tanh activations and a 3D output that represents a correction vector for x, y, z:
- 200 -> 120 -> 60 -> 30 -> 15 -> 3

This architecture is compact and smooth, which helps keep corrections bounded.

## Training Objective (Loss Function)
For each sensor, the network outputs a correction vector. Let x1 and x2 be the raw sensor outputs and y1 and y2 be the predicted corrections. The corrected positions are:

$$
\hat{x}_1 = x_1 + y_1,\quad \hat{x}_2 = x_2 + y_2
$$

The loss enforces that the corrected distance matches the expected separation, plus a small regularization term to prevent drift:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(\|\hat{x}_{1,i}-\hat{x}_{2,i}\| - d_i)^2 + 0.01(\|y_1\|_2^2 + \|y_2\|_2^2)
$$

This ensures:
- Consistency with the known sensor separation.
- Small corrections unless the data demand otherwise.

## Symmetric Training to Avoid Bias
The same network is applied to both sensors. Each sensor is corrected independently, then compared through the distance constraint. This reduces bias that would appear if the model were optimized for only one sensor.

## Training Configuration and Tracking
Key settings from the training pipeline:
- Batch size: 256
- Epochs: 15
- Learning rate: 1e-3
- Validation split: 0.2
- Input size: 200
- Sensor distance label: 0.5 m

Training and validation metrics are tracked with MLflow. The best model is saved and exported to ONNX for deployment.

## Deployment Artifacts
- Web demo for ONNX inference: [notebooks/dsai/phase1_web_nn.pdf](notebooks/dsai/phase1_web_nn.pdf)
- MLflow pipeline and results: [notebooks/dsai/phase2_mlflow_nn.pdf](notebooks/dsai/phase2_mlflow_nn.pdf)
- Streamlit demo for interactive testing: [notebooks/dsai/phase3_streamlit_nn.pdf](notebooks/dsai/phase3_streamlit_nn.pdf)
- MLflow UI snapshots: [notebooks/dsai/mlflow1.pdf](notebooks/dsai/mlflow1.pdf), [notebooks/dsai/mlflow2.pdf](notebooks/dsai/mlflow2.pdf), [notebooks/dsai/mlflow3.pdf](notebooks/dsai/mlflow3.pdf)

## Discussion
### Strengths
- Physics-aware constraint uses the known sensor separation as supervision, avoiding the need for ground-truth global positions.
- Symmetric modeling prevents one sensor from dominating the optimization.
- Windowed inputs allow the network to detect outliers or drift across short time spans.

### Assumptions
- The physical separation between sensors is stable and accurately captured by the distance label.
- Normalization by mean shift removes static bias effectively.
- 50-sample windows capture the relevant temporal context.

### Risks and Limitations
- Distance-only supervision may allow a global shift of both sensors if they drift together.
- If both sensors fail similarly, the constraint may not detect it.
- Regularization strength is heuristic and may need tuning for different operating conditions.

### Potential Improvements
- Add temporal smoothness loss to penalize abrupt corrections between adjacent samples.
- Compare against baseline filters (median or RANSAC) to quantify benefits.
- Try a 1D CNN or attention-based model over the 50-sample window to better capture temporal patterns.
