from __future__ import annotations

from .constants import (
    EPISODE_SUMMARIES_JSON,
    IMG1_KEY,
    IMG2_KEY,
    INPUT_DIR,
    NORM_METHOD,
    REDUCTION_METHOD,
    STATE_KEY,
    TASK_KEY,
)
from .interactive_attention import plot_episode_all_modalities_interactive
from .utils import load_episode_infos
from .series_builder import build_all_series


def default_methods():
    return {
        IMG1_KEY: REDUCTION_METHOD,
        IMG2_KEY: REDUCTION_METHOD,
        TASK_KEY: REDUCTION_METHOD,
        STATE_KEY: REDUCTION_METHOD,
    }


def run_interactive(episode_index=0):
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    methods = default_methods()
    norm = NORM_METHOD

    ep = episodes[episode_index]
    print(f"Opening interactive viewer for Episode {ep.episode_num} | success={ep.success}")

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
    run_interactive(episode_index=0)