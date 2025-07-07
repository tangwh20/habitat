import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2

from omegaconf import OmegaConf
import yaml


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def example():
    # config = habitat.get_config("benchmark/nav/objectnav/objectnav_mp3d.yaml")
    # with open("scripts/example_objnav/example.yaml", "w") as f:
    #     OmegaConf.save(config, f)
    config1 = OmegaConf.load("scripts/example_objnav/example.yaml")
    env = habitat.Env(config=config1)

    print("Environment creation successful")
    observations = env.reset() # keys(['rgb', 'depth', 'objectgoal', 'compass', 'gps'])
    breakpoint()
    # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"])) # Uncomment if using display
    cv2.imwrite("scripts/example_objnav/example_rgb.png", transform_rgb_bgr(observations["rgb"]))

    
    print("Agent stepping around inside environment.")
    print("position: ", env.sim.get_agent_state().position)
    print("rotation: ", env.sim.get_agent_state().rotation)

    count_steps = 0
    keystrokes = [ord(FORWARD_KEY), ord(LEFT_KEY)]
    while not env.episode_over:
        # keystroke = cv2.waitKey(0)
        keystroke = keystrokes[count_steps % 2]

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("position: ", env.sim.get_agent_state().position)
        print("rotation: ", env.sim.get_agent_state().rotation)
        breakpoint()
        # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"])) # Uncomment if using display

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")

        
if __name__ == "__main__":
    example()