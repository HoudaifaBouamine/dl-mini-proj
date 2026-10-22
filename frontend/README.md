# ECG Inspire AI - frontend demo

Nuxt 4 + PrimeVue + Tailwind CSS v4 + Chart.js single-page demo with three tabs
(Signal -> Class, Image -> Signal, Image -> Class). The UI lives in `app/app.vue`.

**Important:** the demo is a UI mock-up. It does not load any trained model. Waveforms are
generated in the browser and the predictions, confidences and "digitised" signals are
hard-coded values returned after a `setTimeout`. See the top-level README.

## Run

Package manager: pnpm (`pnpm-lock.yaml`).

```bash
pnpm install        # runs `nuxt prepare` as postinstall
pnpm dev            # http://localhost:3000
pnpm build          # production build
pnpm preview        # preview the production build
```

Sample images for the "Use Sample" buttons are in `public/samples/`.
