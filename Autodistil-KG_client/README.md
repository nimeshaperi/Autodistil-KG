# Autodistil-KG Pipeline Client

Web UI for the KG Pipeline API: configure pipeline stages, run pipelines, monitor progress, and view results.

Built with **Vite**, **React**, and **TypeScript**. Styled to match the Autodistil-KG Pipeline design (tabs, stage cards, collapsible config sections, progress and logs).

## Run on the same host (default)

With the API running on the same machine, use Vite’s proxy so the client talks to the API without CORS:

1. Start the API (e.g. `uv run uvicorn kg_pipeline_api.main:app --reload` on port 8000).
2. From this directory:
   ```bash
   npm install
   npm run dev
   ```
3. Open http://localhost:5173. The dev server proxies `/api` and `/ws` to the API (see `vite.config.ts`). So the client uses relative URLs and works without setting any env.

## Run on a different client

To run the app from another origin (e.g. another machine, or a static host), point it at your API with **`VITE_API_URL`**:

- **Development:** create a `.env` file:
  ```bash
  VITE_API_URL=http://localhost:8000
  ```
  or the full URL of your API (e.g. `https://api.example.com`). Then `npm run dev`; all requests go to that URL.

- **Production build:** set the variable at build time so it’s baked into the bundle:
  ```bash
  VITE_API_URL=https://your-api.example.com npm run build
  ```
  Serve the `dist` folder with any static server (Nginx, S3, Netlify, etc.). The app will call `https://your-api.example.com` for REST and `wss://your-api.example.com` for WebSockets (if you add WS support later).

No code changes are required; only the env (or build-time) variable changes per deployment.

## Scripts

- `npm run dev` – start dev server (default port 5173)
- `npm run build` – production build to `dist`
- `npm run preview` – serve `dist` locally
- `npm run lint` – run ESLint

## Project layout

- `src/App.tsx` – Tabs and state (configure / monitor / results)
- `src/api/client.ts` – API base URL from `VITE_API_URL`, REST helpers
- `src/components/` – Header, ConfigurePipeline, MonitorProgress, ResultsOutput
- `src/types/config.ts` – Pipeline config and stage types
