# Adaptive Multi-Step Gradient

## Intuition

This optimizer follows the surrogate gradient one accepted step at a time, but
chooses the step size adaptively instead of evaluating a fixed step menu. The
selected rule uses validity-aware backtracking and increases the next step after
clear progress.

## Configuration

| Parameter | Value |
|-----------|-------|
| Step rule | bold-driver backtracking |
| Initial step | probability-gap scaled, clipped to normalized bounds |
| Maximum iterations | 180 |
| Maximum evaluations | 900 |
| Crossing refinement | binary bracketing after the first valid threshold crossing |
| Selection rule | nearest valid crossing after bracketing |
| Oracle use | reporting only |

## Inputs And Outputs

Standard Chapter 5 inputs and schema-v1 outputs.

## Mathematical Description

At iterate \(x_t\), the optimizer computes the MLP probability gradient in
benchmark-normalized coordinates:

$$
d_t = \frac{\nabla_{\tilde{x}} P(y=1\mid x_t)}
{\|\nabla_{\tilde{x}} P(y=1\mid x_t)\|_2}.
$$

It proposes \(x_{t+1}=x_t+\alpha_t d_t\sigma\), where \(\sigma\) is the
benchmark scale vector. If the candidate is invalid or fails to improve the
surrogate probability, \(\alpha_t\) is repeatedly shrunk. After accepted progress,
the next step is expanded up to a local cap. Once a valid candidate crosses
\(P(y=1\mid x)\ge\tau\), binary bracketing between the previous valid point and
the crossing reduces the selected distance.

## Failure Modes

- The local MLP gradient can point toward dense parameter changes.
- Backtracking may still stall when the surrogate gradient aims into a validity
  boundary.
- Hard starts may not cross the threshold within the evaluation budget.

## Thesis Interpretation

The method tests whether true adaptive step-size control can match the robustness
of the multi-fixed-step gradient while avoiding a discrete fixed step menu.
