#!/bin/zsh
# Enhanced OpenModelica Docker setup for macOS with XQuartz
# Add this to your .zshrc file

# Function to launch OpenModelica with proper X11 setup
openmodelica-gui() {
    # Ensure XQuartz is running
    if ! pgrep -x XQuartz > /dev/null; then
        echo "Starting XQuartz..."
        open -a XQuartz
        sleep 3
    fi

    # Configure XQuartz to accept connections - disable access control temporarily
    echo "Configuring X11 access..."
    xhost +

    # Kill any existing socat process on port 6000
    lsof -ti:6000 | xargs kill -9 2>/dev/null

    # Start socat to forward X11 connections from TCP to Unix socket
    echo "Starting X11 forwarding bridge..."
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:/tmp/.X11-unix/X0 &
    local SOCAT_PID=$!
    sleep 1

    # Use host.docker.internal for Docker Desktop on macOS
    local DISPLAY_VAR="host.docker.internal:0"
    echo "Starting OpenModelica GUI (DISPLAY=$DISPLAY_VAR)..."

    # Run OpenModelica Docker container with X11 authorization
    docker run -it --rm \
        -v "$HOME:$HOME" \
        -v "$HOME/.Xauthority:/home/docker/.Xauthority:rw" \
        -e "HOME=$HOME" \
        -e "XAUTHORITY=/home/docker/.Xauthority" \
        -w "$PWD" \
        -e "DISPLAY=$DISPLAY_VAR" \
        --network host \
        --user $UID \
        openmodelica/openmodelica:v1.25.4-gui \
        OMEdit

    # Clean up
    echo "Cleaning up X11 forwarding..."
    kill $SOCAT_PID 2>/dev/null

    # Re-enable X11 access control
    xhost -
}

# Keep your original alias for command-line usage
alias docker-om='docker run -it --rm -v "$HOME:$HOME" -e "HOME=$HOME" -w "$PWD" --user $UID openmodelica/openmodelica:v1.25.4-gui'

# New alias that uses the function
alias omgui='openmodelica-gui'
