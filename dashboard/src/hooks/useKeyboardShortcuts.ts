import { useEffect } from "react";

interface ShortcutHandlers {
  onApprove?: () => void;
  onOverride?: () => void;
  onTabChange?: (tab: "operations" | "scenario" | "kpis" | "results") => void;
  onEscape?: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Don't trigger if user is typing in an input or textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        if (e.key === "Escape") {
          handlers.onEscape?.();
        }
        return;
      }

      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        handlers.onApprove?.();
      } else if (e.key === "o" || e.key === "O") {
        e.preventDefault();
        handlers.onOverride?.();
      } else if (e.key === "Escape") {
        e.preventDefault();
        handlers.onEscape?.();
      } else if (e.key === "1") {
        handlers.onTabChange?.("operations");
      } else if (e.key === "2") {
        handlers.onTabChange?.("scenario");
      } else if (e.key === "3") {
        handlers.onTabChange?.("kpis");
      } else if (e.key === "4") {
        handlers.onTabChange?.("results");
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handlers]);
}
