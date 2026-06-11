import collections
import dataclasses
import json
import logging
import math
import pathlib
import re
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_episodes: Optional[int] = None  # If set, select this many episodes total, balanced across base
    # tasks, perturbation categories, and difficulty levels (per task_classification.json)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


# Variation suffixes appended to LIBERO-Plus task names on top of the base LIBERO-10/Spatial/etc.
# task (e.g. "..._table_3", "..._view_0_0_100_0_0_initstate_12", "..._light_5", "..._language_2").
_VARIATION_SUFFIX_RE = re.compile(
    r"(_view_[\d_-]+_initstate_\d+|_(table|tb)_\d+|_initstate_\d+|_level\d+_sample\d+|_add_\d+|_light_\d+|_noise_\d+|_language_\d+)$"
)


def _base_task_name(name: str) -> str:
    """Strips LIBERO-Plus variation suffixes to recover the underlying base task name."""
    while True:
        new_name = _VARIATION_SUFFIX_RE.sub("", name)
        if new_name == name:
            return name
        name = new_name


def _load_task_classification() -> dict:
    path = pathlib.Path(benchmark.__file__).parent / "task_classification.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _get_task_metadata(task_suite, suite_name: str) -> list[dict]:
    """Returns per-task (base, category, difficulty) metadata used for balanced sampling.

    Falls back to a single category/difficulty bucket (keyed by the task's own name) when the
    suite has no entries in task_classification.json (e.g. libero_90).
    """
    classification = _load_task_classification().get(suite_name, [])
    by_name = {entry["name"]: entry for entry in classification}
    metadata = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        entry = by_name.get(task.name)
        if entry is None:
            metadata.append({"base": task.name, "category": "default", "difficulty": 0})
        else:
            metadata.append(
                {
                    "base": _base_task_name(entry["name"]),
                    "category": entry["category"],
                    "difficulty": entry["difficulty_level"],
                }
            )
    return metadata


def _select_episode_counts(metadata: list[dict], max_episodes: int, capacity: int, seed: int) -> np.ndarray:
    """Greedily distributes `max_episodes` episode slots across tasks so the running selection
    stays as balanced as possible across base tasks, perturbation categories, and difficulty
    levels simultaneously. Returns a per-task episode count (each <= capacity).
    """
    n_tasks = len(metadata)
    bases = sorted({m["base"] for m in metadata})
    categories = sorted({m["category"] for m in metadata})
    difficulties = sorted({m["difficulty"] for m in metadata})

    base_ids = np.array([bases.index(m["base"]) for m in metadata])
    cat_ids = np.array([categories.index(m["category"]) for m in metadata])
    diff_ids = np.array([difficulties.index(m["difficulty"]) for m in metadata])

    base_count = np.zeros(len(bases))
    cat_count = np.zeros(len(categories))
    diff_count = np.zeros(len(difficulties))

    remaining = np.full(n_tasks, capacity, dtype=int)
    counts = np.zeros(n_tasks, dtype=int)

    rng = np.random.default_rng(seed)
    jitter = rng.random(n_tasks) * 1e-6  # tiny fixed-per-task tie-break, avoids always picking task 0

    n_select = min(max_episodes, int(remaining.sum()))
    for _ in range(n_select):
        score = (
            base_count[base_ids] / len(bases)
            + cat_count[cat_ids] / len(categories)
            + diff_count[diff_ids] / len(difficulties)
            + jitter
        )
        score = np.where(remaining > 0, score, np.inf)
        best = int(np.argmin(score))

        counts[best] += 1
        remaining[best] -= 1
        base_count[base_ids[best]] += 1
        cat_count[cat_ids[best]] += 1
        diff_count[diff_ids[best]] += 1

    return counts


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    video_out_path = pathlib.Path(args.video_out_path)
    video_out_path.mkdir(parents=True, exist_ok=True)

    # episode summary bookkeeping
    episode_summaries = []
    infer_global_idx = 0  # increments once per client.infer(...), matching step_*.npy numbering

    # Pre-compute how many episodes to run per task. If max_episodes is set, distribute the
    # episode budget so that base tasks, perturbation categories, and difficulty levels (per
    # task_classification.json) are all covered as evenly as possible. Otherwise run
    # num_trials_per_task episodes for every task, as before.
    if args.max_episodes is not None:
        task_metadata = _get_task_metadata(task_suite, args.task_suite_name)
        episode_counts = _select_episode_counts(
            task_metadata, args.max_episodes, args.num_trials_per_task, args.seed
        )
    else:
        episode_counts = np.full(num_tasks_in_suite, args.num_trials_per_task, dtype=int)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if episode_counts[task_id] == 0:
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Compute which episode indices to run for this task
        n_available = len(initial_states)
        episode_indices = np.linspace(0, n_available - 1, min(episode_counts[task_id], n_available), dtype=int)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(episode_indices):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            start_idx = infer_global_idx
            end_idx = infer_global_idx - 1

            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        end_idx = infer_global_idx
                        infer_global_idx += 1
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            episode_summaries.append(
                {
                    "episode_num": int(episode_idx),
                    "task_id": int(task_id),
                    "task": str(task_description),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "success": bool(done),
                }
            )

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_out_path / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    out_path = video_out_path / "episode_summaries.json"
    with open(out_path, "w") as f:
        json.dump(episode_summaries, f, indent=2)
    logging.info(f"Saved episode summaries to: {out_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
