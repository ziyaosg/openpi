from __future__ import annotations

from .config import (
    EPISODE_NUM,
    EPISODE_SUMMARIES_JSON,
    IMG1_KEY,
    IMG2_KEY,
    INPUT_DIR,
    NORM_METHOD,
    REDUCTION_METHOD,
    STATE_KEY,
    TASK_ID,
    TASK_KEY,
)
from .interactive_attention import plot_episode_all_modalities_interactive
from ...attention_utils.io import load_episode_infos
from .series_builder import build_all_series


def default_methods():
    return {
        IMG1_KEY: REDUCTION_METHOD,
        IMG2_KEY: REDUCTION_METHOD,
        TASK_KEY: REDUCTION_METHOD,
        STATE_KEY: REDUCTION_METHOD,
    }


def find_episode(episodes, task_id, episode_num):
    matches = [ep for ep in episodes if ep.task_id == task_id and ep.episode_num == episode_num]
    if not matches:
        available = sorted({(ep.task_id, ep.episode_num) for ep in episodes})
        raise ValueError(
            f"No episode with task_id={task_id}, episode_num={episode_num}. "
            f"Available (task_id, episode_num) pairs: {available}"
        )
    return matches[0]


def run_interactive():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    methods = default_methods()
    norm = NORM_METHOD

    ep = find_episode(episodes, task_id=TASK_ID, episode_num=EPISODE_NUM)
    print(f"Opening interactive viewer for task_id={ep.task_id}, episode_num={ep.episode_num} | success={ep.success}")

    all_series = build_all_series(
        ep=ep,
        input_dir=INPUT_DIR,
        methods=methods,
        norm=norm,
    )

    plot_episode_all_modalities_interactive(
        ep=ep,
        input_dir=INPUT_DIR,
        all_series=all_series,
        methods=methods,
        norm=norm,
    )


if __name__ == "__main__":
    run_interactive()
