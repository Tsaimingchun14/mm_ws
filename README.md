# mm_ws

## Joint controller
```
ros2 run joint_controller qp_servo_piper_node --ros-args --remap /joint_commands:=/joint_states
```

```
ros2 topic pub /target_ee_pose geometry_msgs/Pose "{position: {x: 0.2 y: 0.1, z: 0.2}, orientation: {w: 0.0, x: 0.0, y: 1.0, z: 0.0}}"
```