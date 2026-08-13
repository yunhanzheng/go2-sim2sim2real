#!/usr/bin/env bash
# Verify the Mac is receiving live state from the Go2 over Ethernet.
# Run in a shell where you've already done:  source deploy_mac/env.sh
#
# Prereq: Go2 powered on and connected to en13 (ping 192.168.123.161 should work).

set -e
echo "Mac IP/iface check:"
route get 192.168.123.161 2>/dev/null | grep -E "interface" || echo "  (no route to robot - is it connected?)"

echo
echo "Available ROS 2 topics (expect /lowstate among them):"
ros2 topic list

echo
echo "Echoing one /lowstate sample (Ctrl-C to stop if it streams)..."
ros2 topic echo --once /lowstate unitree_go/msg/LowState
