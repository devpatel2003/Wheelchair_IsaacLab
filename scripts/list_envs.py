# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `WheelchairRL` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import WheelchairRL.tasks  # noqa: F401
from WheelchairRL.tasks.direct.wheelchairrl.aruco_task_env import ArucoTaskEnv
from WheelchairRL.tasks.direct.wheelchairrl.aruco_task_env_cfg import ArucoTaskEnvCfg

def main():
    """Print all environments registered in `WheelchairRL` extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in WheelchairRL"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        # Filter only tasks that come from the WheelchairRL package
        if "Wheelchairrl" in task_spec.id or "wheelchairrl" in str(task_spec.entry_point):
            cfg = task_spec.kwargs.get("env_cfg_entry_point", "N/A")
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, cfg])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
