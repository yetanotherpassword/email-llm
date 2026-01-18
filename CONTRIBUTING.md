# Contributing to Email-LLM

Thank you for your interest in contributing to Email-LLM! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/email-llm.git
   cd email-llm
   ```
3. Set up the development environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting

Before submitting a PR, please run:

```bash
black src/ tests/
ruff check src/ tests/
```

### Running Tests

```bash
pytest
```

### Type Hints

We use type hints throughout the codebase. Please ensure your code includes proper type annotations.

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear, descriptive messages

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request against the `main` branch

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Thunderbird version and installation type (standard, Snap, Flatpak)
- Steps to reproduce
- Expected vs actual behavior
- Any relevant error messages

## Feature Requests

Feature requests are welcome! Please open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to build something useful together.

## Questions?

Feel free to open an issue for any questions about contributing.
