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
    # config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    # with open("scripts/example/example.yaml", "w") as f:
    #     OmegaConf.save(config, f)
    config1 = OmegaConf.load("scripts/example/example.yaml")
    env = habitat.Env(config=config1)
    # breakpoint()

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    cv2.imwrite("scripts/example/example_rgb.png", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    breakpoint()
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

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        print("position: ", env.sim.get_agent_state().position)
        print("rotation: ", env.sim.get_agent_state().rotation)
        breakpoint()
        # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

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