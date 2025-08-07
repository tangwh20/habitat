import json
import gzip
import quaternion
import numpy as np
from typing import TYPE_CHECKING, cast, Union

import habitat
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.agent import Agent

if TYPE_CHECKING:
    from habitat.core.simulator import Observations, AgentState
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class FixedAgent(Agent):
    """
    An agent that follows a fixed sequence of actions.
    
    Args:
        task_name (str): The name of the task, either "vlnce" or "objectnav".
        data_path (str): Path to the data file containing ground truth actions. (Only required for "vlnce" task)
    """
    def __init__(self, task_name: str, data_path: str = None):
        assert task_name in ["vlnce", "objectnav"], "Invalid task name"
        self.task_name = task_name
        if self.task_name == "vlnce":
            assert data_path is not None, "Data path must be provided for VLNCE task"
            with gzip.open(data_path, "rt") as f:
                self.data = json.load(f)

        self.actions = None
        self.episode_id = None

        self.views = []
        self.positions = []
        self.rotations = []

    def act(self, observations, current_step):
        if current_step < len(self.actions):
            action = self.actions[current_step]
            self.views.append(observations["rgb"])
        else:
            action = 0
        return action

    def reset(self, episode: "NavigationEpisode"):
        self.episode_id = str(episode.episode_id)
        if self.task_name == "vlnce":
            self.actions = np.array(self.data[self.episode_id]["actions"])
        else:
            shortest_path = episode.shortest_paths[0]
            actions = [point.action for point in shortest_path]
            self.actions = np.array(actions)
        self.views.clear()
        self.positions.clear()
        self.rotations.clear()
    
    def update_state(self, agent_state: "AgentState"):
        position = agent_state.position.tolist()
        self.positions.append(position)

        rotation = quaternion.as_float_array(agent_state.rotation).tolist()
        self.rotations.append(rotation)

    # @property
    # def waypoints(self):
    #     return np.array(self.data[self.episode_id]["locations"])
    

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
            stop_on_error=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass

    @property
    def state(self):
        return self.env.sim.get_agent_state()