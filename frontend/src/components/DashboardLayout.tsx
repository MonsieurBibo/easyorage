import { Outlet } from "@tanstack/react-router"
import { DashboardHeader } from "./dashboard/header"

export function DashboardLayout() {
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground md:h-screen">
      <DashboardHeader />
      <main className="flex-1 overflow-auto p-3 pt-2 md:p-4 md:pt-3">
        <Outlet />
      </main>
    </div>
  )
}
