import tailwindcss from '@tailwindcss/vite'

export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: [
    '@primevue/nuxt-module'
  ],
  vite: {
    plugins: [
      tailwindcss()
    ]
  },
  css: [
    '~/assets/css/main.css',
    'primeicons/primeicons.css'
  ],
  primevue: {
    options: {
      theme: 'none' // We can load custom preset or unstyled
    }
  }
})
