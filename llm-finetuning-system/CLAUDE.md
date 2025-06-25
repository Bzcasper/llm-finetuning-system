# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive LLM Fine-Tuning Studio - a professional-grade platform for training Large Language Models using modern techniques like LoRA and QLoRA. The system consists of:

- **Backend**: FastAPI server with Modal.com integration for cloud GPU training
- **Frontend**: Next.js React application with shadcn/ui components
- **Training Engine**: Modal.com serverless functions for distributed fine-tuning
- **Storage**: Multi-backend support (Modal volumes, MinIO, S3)

## Architecture

### Backend Structure
- `api_server.py` - Main FastAPI application with CORS, training job management, and Modal integration
- `finetune_llm.py` - Modal.com functions for distributed fine-tuning with LoRA/QLoRA
- `api/` - Modular API endpoints (health, training, datasets, modal-status)
- Training jobs are managed in-memory (production should use database)
- Real-time monitoring with polling-based status updates

### Frontend Structure  
- Next.js app in `frontend/` directory with TypeScript support
- Uses shadcn/ui component library and Tailwind CSS
- Multi-tab interface: Configuration, Dataset, Training, Monitoring, Credentials
- Real-time training visualization with Recharts
- API communication via fetch with 2-second polling intervals

### Modal Integration
- Cloud functions deployed to "llm-finetuner" app with "ai-tool-pool" environment
- GPU-accelerated training with configurable types (T4, A100, H100, etc.)
- Automatic model/dataset downloading from HuggingFace or storage backends
- Built-in monitoring and resource tracking

## Common Development Commands

### Backend Development
```bash
# Start local API server
python api_server.py

# Deploy Modal functions
export MODAL_PROFILE=ai-tool-pool
modal deploy finetune_llm.py

# Install backend dependencies
pip install -r requirements.txt
```

### Frontend Development
```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build
npm run start

# Linting and testing
npm run lint
npm run test
npm run test:watch

# Database operations (Prisma)
npm run db:generate
npm run db:push
npm run db:migrate
npm run db:studio
```

### Full System Deployment
```bash
# Run complete deployment script
chmod +x deploy.sh
./deploy.sh
```

## Key Configuration Files

- `requirements.txt` - Minimal backend dependencies (Modal, FastAPI, etc.)
- `frontend/package.json` - Frontend dependencies and scripts
- `finetune_llm.py:finetune_image` - Complete ML dependencies (installed in Modal container)
- `.env.example` - Environment variables template
- `frontend/prisma/schema.prisma` - Database schema for user management

## Modal.com Integration Details

### Authentication
- Requires Modal token configuration: `modal token set --token-id ID --token-secret SECRET`
- Uses "ai-tool-pool" profile for deployment
- HuggingFace secrets configured via Modal secrets

### Storage Patterns
- Primary: Modal volumes (`/vol/`) for persistent storage
- Secondary: MinIO and S3 support via `StorageManager` class
- Dataset loading: HuggingFace datasets, local uploads, or volume storage
- Model saving: Automatic upload to configured storage backend

### GPU Configuration
- Configurable GPU types in training config
- Automatic resource allocation based on model size
- 4-bit quantization (QLoRA) support for memory efficiency
- Real-time GPU monitoring via GPUtil and psutil

## Development Workflow

1. **Local Development**: Use `python api_server.py` and `npm run dev` for frontend
2. **Modal Functions**: Deploy with `modal deploy finetune_llm.py` 
3. **Testing**: API integration tests in `tests/` directory
4. **Frontend Testing**: Jest and React Testing Library setup

## Architecture Considerations

- Backend serves as orchestrator, actual training runs on Modal cloud
- In-memory job storage (should be replaced with database for production)
- Real-time updates via polling (consider WebSockets for production)
- Modular API design allows for easy extension
- Storage abstraction supports multiple backends
- Component-based frontend with reusable UI elements

## Security Notes

- API keys and tokens stored in Modal secrets
- CORS configured for development (restrict origins in production)
- Password input fields for sensitive credentials
- Environment variable configuration for deployment settings