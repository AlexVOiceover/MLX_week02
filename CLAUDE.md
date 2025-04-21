# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `pip install -r requirements.txt`
- Run training pipeline (CBOW): `python src/training/word2vec_pipeline.py --model_type cbow_softmax`
- Activate virtual environment: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)

## Workflow Instructions

- When asked for changes or new features, first explain the implementation approach
- Only implement changes after the approach has been discussed and agreed upon
- Provide clear explanations of design decisions and implementation details

## Code Style Guidelines

- **Imports**: Group standard library imports first, followed by third-party packages, then local modules
- **Formatting**: Use 4 spaces for indentation, maximum line length of 88 characters
- **Types**: Use type hints for function parameters and return values
- **Naming**: 
  - Use snake_case for functions and variables
  - Use CamelCase for classes
  - Use lowercase with underscores for file names
- **Documentation**: Docstrings should follow Google style format
- **Error Handling**: Use specific exception types, include appropriate error messages
- **Dataset Classes**: Should implement `__len__` and `__getitem__` methods when inheriting from torch.utils.data.Dataset
- **Model Structure**: Implement modular components for preprocessing, dataset creation, and training