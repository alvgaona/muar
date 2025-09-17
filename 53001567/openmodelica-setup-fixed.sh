#!/bin/zsh
# Enhanced OpenModelica Docker setup for macOS with XQuartz - Fixed version
# Addresses OpenGL issues and improves process management

# Function to launch OpenModelica with proper X11 setup
openmodelica-gui() {
    # Set up signal trap for clean exit
    trap 'cleanup_x11' INT TERM EXIT
    
    # Function for cleanup
    cleanup_x11() {
        echo "\nCleaning up X11 forwarding..."
        if [[ -n "$SOCAT_PID" ]] && kill -0 $SOCAT_PID 2>/dev/null; then
            kill $SOCAT_PID 2>/dev/null
            wait $SOCAT_PID 2>/dev/null
        fi
        xhost - >/dev/null 2>&1
        trap - INT TERM EXIT
    }
    
    # Ensure XQuartz is running (check for X11.bin or Xquartz process)
    if ! pgrep -x "X11.bin" > /dev/null && ! pgrep -x "Xquartz" > /dev/null; then
        echo "Starting XQuartz..."
        open -a XQuartz
        # Wait for XQuartz to fully start
        local wait_count=0
        while ! pgrep -x "X11.bin" > /dev/null && ! pgrep -x "Xquartz" > /dev/null && [[ $wait_count -lt 30 ]]; do
            sleep 0.5
            ((wait_count++))
        done
        if [[ $wait_count -eq 30 ]]; then
            echo "Error: XQuartz failed to start"
            return 1
        fi
        sleep 2  # Give XQuartz time to initialize
    else
        echo "XQuartz is already running."
    fi

    # Configure XQuartz to accept connections - disable access control temporarily
    echo "Configuring X11 access..."
    xhost + >/dev/null 2>&1
    
    # Set XQuartz preferences for better window management
    defaults write org.xquartz.X11 app_to_run /usr/bin/true
    defaults write org.xquartz.X11 no_auth 1
    defaults write org.xquartz.X11 nolisten_tcp 0
    
    # Kill any existing socat process on port 6000
    lsof -ti:6000 2>/dev/null | xargs kill -9 2>/dev/null

    # Start socat to forward X11 connections from TCP to Unix socket
    echo "Starting X11 forwarding bridge..."
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:/tmp/.X11-unix/X0 &
    local SOCAT_PID=$!
    sleep 1
    
    # Verify socat is running
    if ! kill -0 $SOCAT_PID 2>/dev/null; then
        echo "Error: Failed to start X11 forwarding bridge"
        cleanup_x11
        return 1
    fi

    # Use host.docker.internal for Docker Desktop on macOS
    local DISPLAY_VAR="host.docker.internal:0"
    echo "Starting OpenModelica GUI (DISPLAY=$DISPLAY_VAR)..."
    
    # Set up XDG runtime directory
    local XDG_RUNTIME_DIR="/tmp/runtime-$UID"
    mkdir -p "$XDG_RUNTIME_DIR"
    
    # Start a background process to activate XQuartz windows after a delay
    (sleep 3 && osascript -e 'tell application "XQuartz" to activate' && 
     echo "Activating XQuartz window...") &
    local ACTIVATE_PID=$!
    
    # Run OpenModelica Docker container with X11 authorization and OpenGL fixes
    docker run -it --rm \
        -v "$HOME:$HOME" \
        -v "$HOME/.Xauthority:/home/docker/.Xauthority:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -e "HOME=$HOME" \
        -e "XAUTHORITY=/home/docker/.Xauthority" \
        -e "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
        -e "LIBGL_ALWAYS_INDIRECT=1" \
        -e "LIBGL_ALWAYS_SOFTWARE=1" \
        -e "MESA_GL_VERSION_OVERRIDE=4.5" \
        -e "MESA_GLSL_VERSION_OVERRIDE=450" \
        -e "QT_X11_NO_MITSHM=1" \
        -e "QT_AUTO_SCREEN_SCALE_FACTOR=1" \
        -e "_X11_NO_MITSHM=1" \
        -e "_MITSHM=0" \
        -w "$PWD" \
        -e "DISPLAY=$DISPLAY_VAR" \
        --network host \
        --user $UID \
        --ipc=host \
        openmodelica/openmodelica:v1.25.4-gui \
        sh -c "OMEdit --geometry 1400x900+100+50 || OMEdit"

    local exit_code=$?
    
    # Kill the activation background process if still running
    kill $ACTIVATE_PID 2>/dev/null
    
    # Clean up
    cleanup_x11
    
    return $exit_code
}

# Keep your original alias for command-line usage
alias docker-om='docker run -it --rm -v "$HOME:$HOME" -e "HOME=$HOME" -w "$PWD" --user $UID openmodelica/openmodelica:v1.25.4-gui'

# New alias that uses the function
alias omgui='openmodelica-gui'

# Export the function so it's available in subshells
export -f openmodelica-gui