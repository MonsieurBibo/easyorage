// Simple logging utility for the frontend
// In production, this could be replaced with a proper logging service

type LogLevel = "debug" | "info" | "warn" | "error"

interface Logger {
  debug: (...args: unknown[]) => void
  info: (...args: unknown[]) => void
  warn: (...args: unknown[]) => void
  error: (...args: unknown[]) => void
}

const shouldLog = import.meta.env.DEV

function formatMessage(level: LogLevel, ...args: unknown[]): string {
  const timestamp = new Date().toISOString()
  const message = args.map((arg) => {
    if (arg instanceof Error) {
      return arg.message
    }
    if (typeof arg === "object" && arg !== null) {
      try {
        return JSON.stringify(arg)
      } catch {
        return String(arg)
      }
    }
    return String(arg)
  }).join(" ")

  return `[${timestamp}] [${level.toUpperCase()}] ${message}`
}

export const logger: Logger = {
  debug: (...args) => {
    if (shouldLog) {
      console.debug(formatMessage("debug", ...args))
    }
  },
  info: (...args) => {
    if (shouldLog) {
      console.info(formatMessage("info", ...args))
    }
  },
  warn: (...args) => {
    if (shouldLog) {
      console.warn(formatMessage("warn", ...args))
    }
  },
  error: (...args) => {
    // Always log errors, even in production
    console.error(formatMessage("error", ...args))
  },
}
