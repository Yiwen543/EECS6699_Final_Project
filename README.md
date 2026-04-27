EECS 6699 Final Project: The Expressive Power of Depth

Columbia University | Department of Electrical Engineering

Course: Mathematics of Deep Learning (EECSE 6699)

🚀 Project Overview

This project empirically validates the Depth Separation Theorem, a cornerstone of deep learning theory. By utilizing iterated sawtooth functions, we demonstrate that deep neural networks possess an exponential advantage over shallow networks in representing high-frequency oscillatory functions under a fixed parameter budget.

The study confirms that the compositional nature of the ReLU activation function allows depth to create complexity that width alone cannot replicate.

🧠 Theoretical Background

1. The Compositional Power of ReLU

The ReLU activation function, $\sigma(x) = \max(0, x)$, is piecewise linear.

A 2-layer network can represent a "mirror" operator $f(x) = |2x - 1|$, creating one peak.

Composing this operator $L$ times results in $2^L$ linear segments (peaks).

Mathematically, the complexity (number of linear regions) grows exponentially with depth but only linearly with width.

2. Depth Separation (Telgarsky, 2016)

For a function with $2^k$ oscillations:

A Deep Network requires only $O(k)$ layers and constant width to achieve perfect reconstruction.

A Shallow Network (e.g., 2-layer) requires $\Omega(2^k)$ neurons to achieve comparable accuracy.

🧪 Experimental Setup

To ensure a fair comparison, we used a parameter-matched design:

Target Function: An iterated sawtooth function with $2^4 = 16$ peaks.

Deep-Narrow Model:

9 layers, width 4.

Parameters: ~80.

Shallow-Wide Model:

2 layers, width 26.

Parameters: ~80.

📊 Key Results

Metric

Deep Model (L=9)

Shallow Model (L=2)

Parameters

81

79

Approximation Error

Near Zero

High (Oscillations missed)

Representation

Perfectly captures all 16 peaks.

Smooths out high-frequency details.

Conclusion: Depth is a fundamental requirement for representing complex, oscillatory logic. The shallow model fails to generate enough linear regions to fit the target, proving that width is not a substitute for depth in terms of expressivity efficiency.

📂 Repository Structure

depth_separation_study.ipynb: Main Jupyter Notebook with code, training loops, and plots.

paper.tex: LaTeX source for the formal research report.

presentation_outline.md: Logical outline for the final project defense.

🛠️ Installation & Usage

Prerequisites

Python 3.10+

PyTorch, NumPy, Matplotlib

Running the Project

# Clone the repository
git clone [https://github.com/Yiwen543/EECS6699_Final_Project.git](https://github.com/Yiwen543/EECS6699_Final_Project.git)

# Install dependencies
pip install torch matplotlib numpy

# Run the experiment
jupyter lab depth_separation_study.ipynb


📚 References

Telgarsky, M. (2016). Benefits of depth in neural networks. COLT.

Eldan, R., & Shamir, O. (2016). The power of depth for feedforward neural networks.

Prof. Predrag R. Jelenković. EECS 6699 Course Material, Columbia University.