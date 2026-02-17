# AI SDK Monorepo

> The world's most comprehensive AI SDK — multi-provider, multi-agent, production-ready.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](sdk/LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9+-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20+-green.svg)](https://nodejs.org/)
[![Tests](https://img.shields.io/badge/Tests-371%20passing-brightgreen.svg)]()

## Overview

This monorepo contains the AI SDK, a comprehensive TypeScript SDK for building AI-powered applications. The SDK provides a unified interface across multiple LLM providers, agent orchestration, memory management, RAG pipelines, and more.

## Quick Links

- [SDK Documentation](./sdk/README.md)
- [Product Requirements Document](./PRD-v1.md)
- [Verification Report](./VERIFICATION_REPORT.md)
- [Architecture Decision Records](./ADRs/)

## Packages

| Package | Description |
|---|---|
| `@ai-sdk/core` | Essential primitives: types, streaming, generateText, streamText |
| `@ai-sdk/providers` | LLM providers: OpenAI, Anthropic, Google Gemini, OpenRouter, Puter.js |
| `@ai-sdk/agents` | Agent system: agent loop, subagents, orchestration patterns |
| `@ai-sdk/tools` | Tool system: definitions, execution, MCP client & server |
| `@ai-sdk/memory` | Memory: 4-tier system, compaction, semantic search |
| `@ai-sdk/rag` | RAG: chunking, vector store, retrieval, hybrid search |
| `@ai-sdk/voice` | Voice: WebSocket streaming, STT/TTS, VAD |
| `@ai-sdk/compliance` | Compliance: PII detection, GDPR, audit logging |
| `@ai-sdk/edge` | Edge: runtime detection, edge-compatible wrappers |
| `@ai-sdk/cli` | CLI: init, generate, provider management, evaluation |
| `@ai-sdk/devtools` | Devtools: tracing, cost calculation, playground, test generation |
| `@ai-sdk/prompts` | Prompts: registry, versioning, A/B testing, DSPy optimization |
| `@ai-sdk/a2a` | A2A Protocol: agent-to-agent communication |
| `@ai-sdk/infra` | Infrastructure: caching, rate limiting, cost tracking |

## Features

### Multi-Provider Support
- OpenAI (GPT-4o, o1, o1-mini)
- Anthropic (Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3 Haiku)
- Google Gemini (Gemini 1.5 Pro, Gemini 1.5 Flash)
- OpenRouter (access to 100+ models)
- Puter.js (free tier available)

### Agent System
- Multi-agent orchestration
- Subagent delegation
- Supervisor patterns
- Pipeline execution
- Tool integration

### Memory Management
- 4-tier memory system (working, episodic, semantic, procedural)
- Automatic compaction strategies
- Semantic search capabilities
- Vector embeddings support

### RAG Pipeline
- Multiple chunking strategies (fixed, semantic, sentence-based)
- Vector storage and retrieval
- Hybrid search (keyword + semantic)
- Document ingestion pipeline

### Developer Tools
- Execution tracing
- Cost calculation
- Interactive playground
- Test case generation
- Debug sessions

### Compliance
- PII detection and redaction
- GDPR compliance tools
- Audit logging
- Content guardrails

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│         (Agents, Workflows, Chatbots, Apps)                 │
├─────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                      │
│    (Subagents, Multi-Agent Patterns, State Machines)        │
├─────────────────────────────────────────────────────────────┤
│                    CAPABILITY LAYER                         │
│       (Skills, Tools, MCP, Memory, RAG, Prompts)            │
├─────────────────────────────────────────────────────────────┤
│                    FOUNDATION LAYER                         │
│   (LLM Providers, Streaming, Tool Calling, Embeddings)      │
├─────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                     │
│      (Observability, Caching, Rate Limiting, Resilience)    │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-org/ai-sdk.git
cd ai-sdk/sdk

# Install dependencies
pnpm install

# Build all packages
pnpm build

# Run all tests
pnpm test
```

## Development

```bash
# Build all packages
pnpm build

# Run all tests
pnpm test

# Run tests for a specific package
pnpm --filter @ai-sdk/agents test

# Lint
pnpm lint

# Format
pnpm format
```

## Testing

The SDK has **371 tests** across **14 packages**, all using real implementations (no mocks):

| Package | Tests |
|---------|-------|
| @ai-sdk/core | 24 |
| @ai-sdk/providers | 35 |
| @ai-sdk/agents | 23 |
| @ai-sdk/memory | 48 |
| @ai-sdk/rag | 8 |
| @ai-sdk/tools | 25 |
| @ai-sdk/compliance | 16 |
| @ai-sdk/prompts | 31 |
| @ai-sdk/devtools | 58 |
| @ai-sdk/cli | 26 |
| @ai-sdk/a2a | 24 |
| @ai-sdk/infra | 41 |
| @ai-sdk/voice | 10 |
| @ai-sdk/edge | 2 |

## Project Structure

```
ai-sdk/
├── sdk/                    # Main SDK packages
│   ├── packages/           # Individual packages
│   ├── package.json        # Root package.json
│   ├── pnpm-workspace.yaml # Workspace configuration
│   └── turbo.json          # Build configuration
├── ADRs/                   # Architecture Decision Records
├── PRD-v1.md               # Product Requirements Document
├── VERIFICATION_REPORT.md  # Verification Report
└── README.md               # This file
```

## Documentation

- [SDK README](./sdk/README.md) - Detailed SDK documentation
- [API Reference](./sdk/docs/api-reference.md) - API documentation
- [ADRs](./ADRs/) - Architecture Decision Records
- [PRD](./PRD-v1.md) - Product Requirements Document
- [Verification Report](./VERIFICATION_REPORT.md) - Codebase verification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](sdk/LICENSE) file for details.

## Support

- GitHub Issues: [https://github.com/your-org/ai-sdk/issues](https://github.com/your-org/ai-sdk/issues)
- Documentation: [https://ai-sdk.dev/docs](https://ai-sdk.dev/docs)
