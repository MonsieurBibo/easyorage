import { Dashboard } from "./components/Dashboard"
import { ThemeProvider } from "@/components/ui/theme-provider"
import { TooltipProvider } from "./components/ui/tooltip"

function App() {
  return (
  <>
  <ThemeProvider defaultTheme="light" storageKey="ui-theme">
    <TooltipProvider>
      <Dashboard />
    </TooltipProvider>
    </ThemeProvider>

  </>
  )
}

export default App
