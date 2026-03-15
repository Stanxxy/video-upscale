# Chrome DevTools MCP Patterns

## Basic Page Verification
```
1. navigate_page → URL
2. take_snapshot → check accessibility tree for expected elements
3. take_screenshot → visual confirmation
```

## Form Testing
```
1. navigate_page → form page
2. take_snapshot → find form fields by role/label
3. fill_form → fill fields using selectors from snapshot
4. click → submit button
5. wait_for → success indicator (text, URL change, element)
6. take_snapshot → verify success state
```

## Responsive Testing
```
1. resize_page → { width: 375, height: 667 }   # Mobile
2. take_screenshot → capture mobile layout
3. resize_page → { width: 768, height: 1024 }   # Tablet
4. take_screenshot → capture tablet layout
5. resize_page → { width: 1280, height: 800 }   # Desktop
6. take_screenshot → capture desktop layout
```

Note: For device-specific testing, `emulate` can simulate specific devices (viewport + user agent) as an alternative to `resize_page`.

## Login Flow
```
1. navigate_page → /login
2. fill_form → [{ selector: "input[name=email]", value: "..." }, { selector: "input[name=password]", value: "..." }]
3. click → submit button (from snapshot ref)
4. wait_for → { text: "Dashboard" } or { url: "/dashboard" }
5. take_snapshot → verify logged-in state
```

## Error State Verification
```
1. navigate_page → page
2. fill_form → invalid data
3. click → submit
4. take_snapshot → check for error messages in tree
5. take_screenshot → visual confirmation of error state
```

## Navigation Testing
```
1. navigate_page → starting page
2. take_snapshot → find nav links
3. click → target link
4. wait_for → { url: "/expected-path" }
5. take_snapshot → verify destination content
```

## Tips
- Always `take_snapshot` before interacting — it gives you accurate selectors
- Use `ref` attributes from snapshot for clicking (most reliable)
- `wait_for` after actions that trigger navigation or async loading
- `list_console_messages` to check for JS errors after interactions
- `list_pages` to verify new tab behavior
