# 0001 – Theme Management and Dark Mode

Date: 2025-11-02

## Status

Accepted

## Context

We introduced a Settings page with a Dark/Light/System theme preference and needed a reliable way to apply a global theme to the PySide6 app and to embedded web views (ECharts, Cytoscape, KaTeX preview). An initial attempt used `QApplication.setStyleSheet` as if it were a static method, causing a runtime error on Windows with PySide6: `TypeError: descriptor 'setStyleSheet' ...`.

## Decision

- Centralize theme logic in a `ThemeManager` singleton that:
  - Stores the user preference (System/Light/Dark) in `settings.json`.
  - Detects system theme heuristically from the current Qt palette.
  - Applies a dark/light `QPalette` and sets a global application stylesheet via the application instance (`QApplication.instance().setStyleSheet(...)`).
  - Emits a `theme_changed` signal used by embedded web views to call their `window.setTheme('light'|'dark')` handlers.
- Update embedded pages (`echart.html`, `cyto.html`, `proof.html`) to expose `window.setTheme(...)` and adapt colors without reloading.
- Apply theme preference on startup before the main window is shown.

## Consequences

- Fixes the runtime error by using instance methods (`app.setStyleSheet`) instead of class-level calls.
- Provides consistent theming between native widgets and embedded web content.
- Users can switch themes live from the Settings page; preference persists across restarts.

## Alternatives Considered

- Using a third-party Qt theming library (qt-material). Rejected for now to keep footprint small and maintain full offline operation; we can revisit later.
- Hard-coding dark-mode styles only in web views. Rejected to ensure native widgets also follow the theme.
- Relying on OS theme only. Rejected to allow explicit Light/Dark overrides for demos and reproducibility.

## Links

- Files: `app/services/theme.py`, `app/ui/pages/settings_page.py`, `app/web/echart.html`, `app/web/cyto.html`, `app/web/proof.html`.

