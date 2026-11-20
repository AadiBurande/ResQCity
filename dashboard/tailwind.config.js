/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        agency: {
          ambulance: "#10b981", // emerald
          police: "#3b82f6",    // blue
          fire: "#ef4444",      // red
        },
        tier: {
          full: "#8b5cf6",      // purple
          verify: "#f59e0b",    // amber
          review: "#ec4899",    // pink
        }
      }
    },
  },
  plugins: [],
}
