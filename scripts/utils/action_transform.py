import numpy as np

STEP_LENGTH = 0.25
TURN_ANGLE = np.pi / 12.

def waypoints_to_actions(waypoints: np.ndarray):
    actions = []
    current_pos = np.array([0., 0.])
    current_yaw = 0.
    idx = 0

    # current_positions = []
    while idx < waypoints.shape[0]:
        # breakpoint()
        target = waypoints[idx]
        distance = np.linalg.norm(target - current_pos)
        if distance < STEP_LENGTH / 2:
            idx += 1
            continue

        target_yaw = np.arctan2(target[1] - current_pos[1], target[0] - current_pos[0])
        turns = int(np.round((target_yaw - current_yaw) / TURN_ANGLE))
        if turns > 0:
            actions.extend([2] * turns)
        elif turns < 0:
            actions.extend([3] * -turns)
        current_yaw += turns * TURN_ANGLE

        if distance > STEP_LENGTH / 2:
            actions.append(1)
            current_pos = current_pos + STEP_LENGTH * np.array([np.cos(current_yaw), np.sin(current_yaw)])
            # current_positions.append(current_pos)
        
        # print(f"current_pos: {current_pos}, current_yaw: {current_yaw},\n \
        #        target: {target}, target_yaw: {target_yaw},\n \
        #        distance: {distance}, \n \
        #        actions: {actions}")
    
    return current_pos, np.array(actions)


def plot_waypoints(waypoints: np.ndarray, filename: str = "waypoints.png"):
    import matplotlib.pyplot as plt
    plt.plot(waypoints[:, 0], waypoints[:, 1], '-o')
    plt.savefig(filename)

# print(waypoints_to_actions(waypoints))
# current_positions, actions = waypoints_to_actions(waypoints)
# plot_waypoints(waypoints, "waypoints.png")
# plot_waypoints(np.array(current_positions), "current_positions.png")
# print(actions)


if __name__ == "__main__":    
    waypoints = np.array([
        [ 0.53103535,  0.10609858],
        [ 1.01612985,  0.33051275],
        [ 1.02712124,  0.00884718],
        [ 1.16489144,  0.2052187 ],
        [ 1.31097756,  0.17370607],
        [ 1.71789087,  0.09104576],
        [ 2.17634428, -0.0257676 ],
        [ 2.54461654,  2.07220189]
    ])
    print(waypoints_to_actions(waypoints))