/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom color scheme for tensor visualization
        tensor: {
          input: '#3b82f6',      // blue-500
          attention: '#10b981',  // green-500
          ffn: '#f59e0b',        // amber-500
          norm: '#8b5cf6',       // violet-500
          output: '#ef4444',     // red-500
          other: '#6b7280',      // gray-500
        },
      },
    },
  },
  plugins: [],
}
