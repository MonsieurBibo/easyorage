import { RouterProvider } from '@tanstack/react-router'
import { ThemeProvider } from "@/components/ui/theme-provider"
import { TooltipProvider } from "./components/ui/tooltip"
import { Toaster } from "./components/ui/sonner"
import { LiveDataProvider } from "@/context/LiveDataContext"
import { ErrorBoundary } from './components/ErrorBoundary'
import { router } from './router'

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="light" storageKey="ui-theme">
        <TooltipProvider>
          <LiveDataProvider>
            <RouterProvider router={router} />
          </LiveDataProvider>
        </TooltipProvider>
      </ThemeProvider>
      <Toaster position="top-right" expand={false} />
    </ErrorBoundary>
  )
}

export default App
