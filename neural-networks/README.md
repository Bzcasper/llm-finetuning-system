# Neural Networks Monorepo

A comprehensive monorepo containing various neural network implementations, optimization techniques, and research materials.

## Structure

This monorepo is organized into the following packages:

- **`common/`** - Shared neural network utilities and base implementations
- **`neural_network_cpu/`** - CPU-optimized neural network implementations
- **`neural_network_optimization/`** - Performance optimization techniques and benchmarks
- **`neural_network_env/`** - Environment setup and configuration
- **`docs/`** - Research documentation and architecture specifications

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 8.0.0
- Python >= 3.8 (for Python packages)

### Installation

```bash
# Install all dependencies across workspaces
npm run install-all

# Or install dependencies for a specific package
cd neural_network_cpu && npm install
```

### Available Scripts

- `npm test` - Run tests across all packages
- `npm run lint` - Run linting across all packages
- `npm run build` - Build all packages
- `npm run clean` - Clean all node_modules and build artifacts

### Packages Overview

#### Common
Shared utilities including:
- Base neural network implementations (`neural_network.py`)
- Loss functions (`losses.py`)
- Optimizers (`optimizers.py`)
- Network architectures (`network.py`)

#### Neural Network CPU
CPU-optimized implementations with:
- Custom activation functions
- Efficient layer implementations
- Comprehensive test suite
- Performance benchmarks

#### Neural Network Optimization
Performance optimization research including:
- CPU optimization techniques
- Data pipeline architecture
- Profiling tools
- Benchmarking frameworks

## Development

### Workspace Management

This monorepo uses npm workspaces for package management. Each package can be developed independently while sharing common dependencies.

### Testing

Each package contains its own test suite. Run tests for all packages:

```bash
npm test
```

Or for a specific package:

```bash
cd neural_network_cpu && npm test
```

## Documentation

See the `docs/` directory for:
- Architecture specifications
- Research findings
- Implementation guides

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see individual packages for specific licensing information.