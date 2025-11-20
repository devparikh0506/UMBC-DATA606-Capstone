# BCI Frontend (React + Vite)

React frontend for BCI real-time prediction application.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will run on `http://localhost:5173`

## Build

To build for production:
```bash
npm run build
```

## Features

- Subject selection
- Run selection (training sessions)
- Real-time prediction visualization with animated smiley
- Results display with accuracy metrics

## Configuration

- API proxy configured in `vite.config.ts` to forward `/api` requests to Django backend
- WebSocket connections go directly to `ws://localhost:8000`

