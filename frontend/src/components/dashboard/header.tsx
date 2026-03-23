import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { ChevronDown } from "lucide-react"
import { ThemeSwitcher } from "@/components/ui/mode-toogle"

export function DashboardHeader() {
  return (
    <header className="border-b border-border bg-card">
      <div className="flex h-16 items-center px-4 md:px-8 gap-8">
        <div className="flex items-center gap-2 px-3 py-1.5 border border-border rounded-md shadow-sm cursor-pointer hover:bg-muted transition-colors">
          <Avatar className="h-5 w-5">
            <AvatarImage src="/ui/Logo_Ville_Bastia.svg" />
            <AvatarFallback>B</AvatarFallback>
          </Avatar>
          <span className="text-sm font-medium text-foreground">Bastia</span>
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        </div>
        
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
          <a href="#" className="text-foreground transition-colors hover:text-primary">Dashboard</a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-primary">Analyses</a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-primary">Historique</a>
          <a href="#" className="text-muted-foreground transition-colors hover:text-primary">Paramètres</a>
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <Avatar className="h-8 w-8">
            <AvatarImage src="https://github.com/shadcn.png" />
            <AvatarFallback>U</AvatarFallback>
          </Avatar>
          <ThemeSwitcher />
        </div>
      </div>
    </header>
  )
}
