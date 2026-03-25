import { Dashboard } from "./components/Dashboard"
import { ThemeProvider } from "@/components/ui/theme-provider"
import { TooltipProvider } from "./components/ui/tooltip"
import { LiveDataProvider } from "@/context/LiveDataContext"

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="ui-theme">
      <TooltipProvider>
        <LiveDataProvider>
          <Dashboard />
        </LiveDataProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
