#!/bin/bash
# Quick performance test - runs simulation and shows key metrics

echo "========================================="
echo "PUCKBOT PERFORMANCE TEST"
echo "========================================="
echo ""

# Run short test match with logging
echo "Running 30-second test match..."
python run.py --mode tournament --game_duration 30 --log

# Find the latest log
LATEST_LOG=$(ls -t logs/tournament_mode/*.json | head -1)

if [ -f "$LATEST_LOG" ]; then
    echo ""
    echo "========================================="
    echo "PERFORMANCE METRICS"
    echo "========================================="
    
    # Extract key metrics using jq (if available) or grep/python
    if command -v jq &> /dev/null; then
        echo ""
        echo "RIGHT ROBOT:"
        echo "  Contacts: $(jq '.robot_stats.right.strikes.contacted' "$LATEST_LOG")"
        echo "  Accuracy: $(jq '.robot_stats.right.strikes.accuracy' "$LATEST_LOG" | awk '{printf "%.1f%%", $1*100}')"
        echo "  Goals: $(jq '.robot_stats.right.offense.goals_scored' "$LATEST_LOG")"
        echo "  Distance: $(jq '.robot_stats.right.movement.total_distance_m' "$LATEST_LOG")m"
        
        echo ""
        echo "LEFT ROBOT:"
        echo "  Contacts: $(jq '.robot_stats.left.strikes.contacted' "$LATEST_LOG")"
        echo "  Accuracy: $(jq '.robot_stats.left.strikes.accuracy' "$LATEST_LOG" | awk '{printf "%.1f%%", $1*100}')"
        echo "  Goals: $(jq '.robot_stats.left.offense.goals_scored' "$LATEST_LOG")"
        echo "  Distance: $(jq '.robot_stats.left.movement.total_distance_m' "$LATEST_LOG")m"
        
        echo ""
        echo "MATCH:"
        echo "  Final Score: $(jq '.final_score.right' "$LATEST_LOG") - $(jq '.final_score.left' "$LATEST_LOG")"
        echo "  Duration: $(jq '.match_info.duration_seconds' "$LATEST_LOG")s"
    else
        echo ""
        echo "Install jq for detailed metrics: brew install jq"
        echo "Raw log: $LATEST_LOG"
        cat "$LATEST_LOG"
    fi
    
    echo ""
    echo "========================================="
fi
