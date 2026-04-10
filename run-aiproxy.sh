#!/bin/bash

SESSION_NAME="aiproxy"

cleanup() {
    tmux kill-session -t $SESSION_NAME 2>/dev/null
} 

trap cleanup EXIT

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session with ngrok
tmux new-session -d -s $SESSION_NAME "ngrok http 11434; read -p 'Press Enter to close...'"

# Split vertically and run aiproxy
tmux split-window -v -t $SESSION_NAME "./aiproxy.py --model-mapping '{\"gpt-5.4\": \"big-pickle\", \"gpt-5.2\": \"nemotron-3-super-free\"}' --coerce-input-to-messages --sanitize-chat-tools $@"

# Attach to session
tmux attach -t $SESSION_NAME