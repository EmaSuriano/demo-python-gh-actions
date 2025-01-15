# Demo Python GitHub Actions

This project demonstrates how to use GitHub Actions to train a machine learning model on the CI and obtain the trained model using artifacts.

## Project Overview

The project uses `uv` (https://docs.astral.sh/uv/), an extremely fast Python package and project manager, written in Rust.

## Goals

- Show how to set up GitHub Actions for continuous integration.
- Train a machine learning model as part of the CI process.
- Use GitHub Actions artifacts to store and retrieve the trained model.

## Getting Started

1. **Install `uv`:**
   Follow the instructions on the [uv documentation](https://docs.astral.sh/uv/) to install `uv`.

2. **Set up the project:**
   Clone the repository and set up the project using `uv`.

```sh
git clone https://github.com/yourusername/demo-python-gh-actions.git
cd demo-python-gh-actions
uv setup
```

3. **Configure GitHub Actions:**
   Ensure that your GitHub repository has the necessary workflows configured to train the model and store the artifacts.

## Usage

To train the model locally, you can use the following command:

```sh
uv run train
```

To retrieve the trained model from the GitHub Actions artifacts, follow the instructions provided in the workflow documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
