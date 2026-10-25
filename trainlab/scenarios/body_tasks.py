"""Body + Task scenarios: every pair listed here becomes a normal TrainLab scenario named
``<body>_<task>`` (trainlab train drone_landing, Studio, live view, resume...).

Add a body or a task, then add the pair here.
"""

from trainlab.bodies.drone import DroneBody
from trainlab.bodies.rover import RoverBody
from trainlab.core.registry import register_scenario
from trainlab.tasks.base import body_task_env
from trainlab.tasks.landing import LandingTask
from trainlab.tasks.waypoints import WaypointsTask

PAIRS = [
    (DroneBody, WaypointsTask),
    (DroneBody, LandingTask),
    (RoverBody, WaypointsTask),
]

for _body, _task in PAIRS:
    register_scenario(f"{_body.name}_{_task.name}")(body_task_env(_body, _task))
