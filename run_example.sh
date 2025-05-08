#!/bin/bash

# Build and start the container
docker-compose up -d

# Check if a specific tool is provided as an argument
if [ $# -eq 0 ]; then
    # If no argument provided, run the interactive example
    docker-compose exec pokerai python examples/interactive_user_v_user.py
else
    # Run the specified tool
    docker-compose exec pokerai python "$@"
fi 