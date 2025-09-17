#!/bin/bash
# Script to activate XQuartz and bring windows to front

# Activate XQuartz application
osascript -e 'tell application "XQuartz" to activate'

# Wait a moment for activation
sleep 0.5

# Try to bring all X11 windows to front
osascript <<EOF
tell application "System Events"
    tell process "XQuartz"
        set frontmost to true
        try
            click menu item "Bring All to Front" of menu "Window" of menu bar 1
        end try
    end tell
end tell
EOF

echo "XQuartz windows should now be visible"