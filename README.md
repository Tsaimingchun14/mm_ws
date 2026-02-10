# mm_ws

## create environment
```
uv venv --python 3.10
source .venv/bin/activate
uv pip sync requirements.txt --cache-dir {CACHE_DIR} 
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 
#optional
export HF_HOME={CACHE_DIR}
```

## Joint controller
```
ros2 run joint_controller qp_servo_piper_node --ros-args --remap /joint_commands:=/joint_states
```

```
ros2 topic pub /target_ee_pose geometry_msgs/Pose "{position: {x: 0.2, y: 0.1, z: 0.2}, orientation: {w: 0.0, x: 0.0, y: 1.0, z: 0.0}}"
```

## Motion planner
```
ros2 run motion_planner motion_planner_node
ros2 topic pub /motion_task std_msgs/msg/String '{ "data": "{\"action\": \"grasp\", \"point\": [640.0, 360.0] }" }'
```
