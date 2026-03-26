import { RouterProvider } from '@tanstack/react-router'
import { ThemeProvider } from "@/components/ui/theme-provider"
import { TooltipProvider } from "./components/ui/tooltip"
import { LiveDataProvider } from "@/context/LiveDataContext"
import { router } from './router'

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="ui-theme">
      <TooltipProvider>
        <LiveDataProvider>
          <RouterProvider router={router} />
        </LiveDataProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
