"""Generic 'Circle' Aigarth Intelligent Tissue Unit (AITU)."""

import logging
import random
import secrets
from pathlib import Path
from time import process_time
from typing import Any

from pydantic import BaseModel, Field
from sqlite_construct import DBReference, DB_SCHEME
from sqlite_kvdb import SQLiteKVDB, SQLiteKVDBError

from .common import random_trit_vector
from .exceptions import AigarthITUError
from .itu_base import AigarthITU, ITUMeta
from .neuron_cl import AITClNeuron

LOG = logging.getLogger(Path(__file__).stem)


class FFCYCLE_END_REASON:
    NO_OUTPUT_ZEROES = "All 'output' neurons got 'non-zero' state"
    TICK_CAP = "Tick cap reached"
    NO_NSTATE_CHANGES = "None of ITU neurons changed their state"


class FFCycleStats(BaseModel):
    """ITU 'feedforward' cycle statistics container."""

    tick_count: int = 0
    end_reason: str = FFCYCLE_END_REASON.TICK_CAP
    duration_total: float = 0.0
    duration_tick: float = 0.0


class ITUReflection(BaseModel):
    """Reflection definition."""

    trits: tuple[int, ...] = Field(frozen=True)
    ffcycle_stats: FFCycleStats = Field(default_factory=FFCycleStats)


class AigarthITUCl(AigarthITU):
    """Generic 'Circle' Aigarth Intelligent Tissue Unit (AITU) definition."""

    FF_CYCLE_CAP_BASE = 1000000

    def __init__(self, itu_group: str, input_bitwidth: int | None = None, output_bitwidth: int | None = None) -> None:
        """Initialize ITU instance."""
        self.lm_prefix = f"{self.__class__.__name__}: "

        itu_meta_kwargs: dict = dict(group=itu_group, type=self.__class__.__name__)
        if input_bitwidth:
            itu_meta_kwargs["input_bitwidth"] = input_bitwidth
        if output_bitwidth:
            itu_meta_kwargs["output_bitwidth"] = output_bitwidth
        self.meta = ITUMeta(**itu_meta_kwargs)

        self.ff_cycle_cap = self.FF_CYCLE_CAP_BASE * self.meta.output_bitwidth

        # ITU default initial (creation-time) configuration:
        #   * structure consists of 'input' and 'output' neurons only
        #   * all neurons have the only 'forward' input link assigned with randomly chosen weight
        #   * 'input' and 'output' neurons are randomly distributed across the 'tissue material'
        #
        # NOTES:
        #   * Number of 'input' and 'output' neurons (not necessarily, but) might be changed during ITU's lifecycle.
        #   * neuron input configurations (number of inputs, their weights and 'input skew') might be changed
        #     (if/as necessary) individually for each neuron during ITU's lifecycle.
        self._neurons_i = [
            AITClNeuron(input_weights=random_trit_vector(size=1)) for _ in range(self.meta.input_bitwidth)
        ]
        self._neurons_o = [
            AITClNeuron(input_weights=random_trit_vector(size=1)) for _ in range(self.meta.output_bitwidth)
        ]
        self._circle = self._neurons_i[:] + self._neurons_o[:]
        random.shuffle(self._circle)

    def mutate(
        self, training_episode: int, training_season: str | None = None, training_dataset_fpath: Path | None = None
    ) -> None:
        """Mutate.

        :param training_episode:        number of training episode (within a training season)
        :param training_season:         training season id. If specified turns off training season id auto-discovery
                                        from a training dataset file.
        :param training_dataset_fpath:  training dataset file path
        """
        LOG.info(
            f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={self.meta.version_string}, "
            f"neurons={len(self._circle)}: "
            f"training_season={training_season or str(training_dataset_fpath)}, {training_episode=} ..."
        )

        # Identify a neuron's input weight to change
        idx_all_weights = []
        for ni, neuron in enumerate(self._circle):
            if len(neuron.input_weights) >= len(self._circle):
                ii_margin_base, ii_margin_odd = divmod(len(neuron.input_weights) - (len(self._circle) - 1), 2)
                ia_idx_start = ii_margin_base + ii_margin_odd
                ia_idx_end = ia_idx_start + (len(self._circle) - 1)
            else:
                ia_idx_start, ia_idx_end = 0, len(neuron.input_weights)

            for wi in range(ia_idx_start, ia_idx_end):
                idx_all_weights.append((ni, wi))

        idx_neuron, idx_weight = secrets.choice(idx_all_weights)

        # Change input weight
        weight_new = self._circle[idx_neuron]._input_weights[idx_weight] + secrets.choice((-1, 1))

        if weight_new in (-1, 0, 1):
            self._circle[idx_neuron]._input_weights[idx_weight] = weight_new
            LOG.info(
                f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={self.meta.version_string}: "
                f"Change neuron's input weight: neuron_index={idx_neuron}, input_index={idx_weight}, "
                f"new_weight={weight_new}, neuron_inputs_total={len(self._circle[idx_neuron].input_weights)}, "
                f"neurons_total={len(self._circle)}"
            )

            zero_input_weights = [iw for iw in self._circle[idx_neuron]._input_weights if iw == 0]
            if len(zero_input_weights) == len(self._circle[idx_neuron]._input_weights):
                if not (self._circle[idx_neuron] in self._neurons_i or self._circle[idx_neuron] in self._neurons_o):
                    # Remove the 'base' neuron, if removal condition met (all input weights are zero, and it's not a
                    # member of input or output neuron groups)
                    del self._circle[idx_neuron]
                    LOG.info(
                        f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={self.meta.version_string}: "
                        f"Remove ineffective neuron: neuron_index={idx_neuron}, neurons_total={len(self._circle)}"
                    )
        else:
            # Spawn a neuron
            # Get a 'spawn model' neuron (the one to be cloned)
            idx_neuron_spawn_model = self.get_nsmodel_idx(idx_neuron, idx_weight)
            neuron_spawn_model = self._circle[idx_neuron_spawn_model]
            # Clone the 'spawn  model' neuron
            neuron_new = neuron_spawn_model.__class__(
                input_weights=neuron_spawn_model.input_weights[:],
                input_skew=neuron_spawn_model.input_skew,
            )
            # Insert (just cloned) new neuron into the 'circle'
            idx_neuron_new = idx_neuron + 1  # Select location adjacent to the 'base' neuron
            self._circle.insert(idx_neuron_new, neuron_new)
            LOG.info(
                f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={self.meta.version_string}: "
                f"Spawn a neuron: neuron_index_new={idx_neuron_new}, neurons_total={len(self._circle)}"
            )

        # Upgrade ITU version
        itu_version_old = self.meta.version_string
        self.meta.set_version(
            training_episode=training_episode,
            training_season=training_season,
            training_dataset_fpath=training_dataset_fpath,
        )
        LOG.info(
            f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={itu_version_old}: "
            f"Upgrade ITU version: version_new={self.meta.version_string}, neurons_total={len(self._circle)}"
        )

        LOG.info(
            f"{self.lm_prefix}Mutate ITU: uuid={self.meta.id}, version={itu_version_old}: "
            f"version_new={self.meta.version_string}, neurons_total={len(self._circle)}: "
            f"training_season={training_season or str(training_dataset_fpath)}, {training_episode=}: OK"
        )

    def get_nsmodel_idx(self, circle_index: int, weights_index: int) -> int:
        """Get 'circle' index of a neuron to serve as a model for spawning a new neuron in the scope of ITU mutation
        procedure.

        :param circle_index:    index of a 'base' mutation neuron in the 'circle' (the one whose input's change
                                initiated spawning a new neuron)
        :param weights_index:   index of a 'base' mutation neuron's input in the 'weights' of the 'base' mutation neuron
                                (the one whose change initiated spawning a new neuron)
        :return:                'model' neuron's 'circle' index
        """
        neuron_base = self._circle[circle_index]
        len_bw_gr = neuron_base.input_split_index

        if weights_index < neuron_base.input_split_index:
            idx_neuron_spawn_model = (circle_index - len_bw_gr + weights_index) % len(self._circle)
        else:
            idx_neuron_spawn_model = (circle_index + 1 + weights_index - len_bw_gr) % len(self._circle)

        return idx_neuron_spawn_model

    def get_neuron_feed(self, circle_index: int) -> tuple[int, ...]:
        """Get 'feed' values for a neuron.

        :param circle_index:    neuron's 'circle' index
        :return:                list of neuron feed values
        """
        n = self._circle[circle_index]

        oi_margin_base, oi_margin_odd = 0, 0
        if len(n.input_weights) >= len(self._circle):
            oi_margin_base, oi_margin_odd = divmod(len(n.input_weights) - (len(self._circle) - 1), 2)

        len_bw_gr = n.input_split_index
        len_fw_gr = len(n.input_weights) - n.input_split_index

        len_bw_gr_active = len_bw_gr - oi_margin_base - oi_margin_odd
        len_fw_gr_active = len_fw_gr - oi_margin_base

        bw_feed_active = [
            self._circle[i % len(self._circle)].state for i in range(circle_index - len_bw_gr_active, circle_index)
        ]
        bw_feed = [0] * (len_bw_gr - len_bw_gr_active) + bw_feed_active

        fw_feed_active = [
            self._circle[i % len(self._circle)].state
            for i in range(circle_index + 1, circle_index + 1 + len_fw_gr_active)
        ]
        fw_feed = fw_feed_active + [0] * (len_fw_gr - len_fw_gr_active)

        feed = bw_feed + fw_feed

        return tuple(feed)

    def feedforward(self, feed: tuple[int, ...] | None = None) -> tuple[tuple[int, ...], FFCycleStats]:
        """Produce a 'forward' value from the 'feed' data.

        :param feed:    set of balanced 'trit' values ([-1,0,+1]) fed to the ITU as input
        :return:        ITU's output (set of balanced 'trit' values) computed from the input values along with some
                        'feedforward' procedure stats
        """
        # A 'None' feed is a valid (default) input; treat it as an empty feed, as the
        # neuron-level feedforward() does, rather than failing on len(None).
        feed = feed or tuple()
        # Reset states of all neurons
        for n in self._circle:
            n.state = 0
        # Assign ITU's 'feed' to initial states of 'input' neurons
        len_init = min(len(feed), self.meta.input_bitwidth)
        for i in range(len_init):
            self._neurons_i[i].state = feed[i]

        # Run 'feedforward' cycle
        ffcycle_end_reason = FFCYCLE_END_REASON.TICK_CAP
        ffcycle_start_timestamp = process_time()
        for tick in range(self.ff_cycle_cap):  # Ticking break condition #2: Ticks upper limit was reached
            no_n_state_changes = True
            no_zero_out_n_states = True
            # Compute 'next' state for all neurons (forward)
            for i, n in enumerate(self._circle):
                # Collect 'feed' for a neuron
                n_feed = self.get_neuron_feed(i)
                # Compute 'forward' for a neuron
                n.feedforward(n_feed)
            # Promote 'forward' to 'state' for all neurons
            for n in self._circle:
                n_state, changed = n.commit_state()
                if changed:
                    no_n_state_changes = False
            # Ticking break condition #3: no neurons have changed their state
            if no_n_state_changes:
                ffcycle_end_reason = FFCYCLE_END_REASON.NO_NSTATE_CHANGES
                break
            # Ticking break condition #1: all 'output' neurons got non-zero state
            for n in self._neurons_o:
                if n.state == 0:
                    no_zero_out_n_states = False
                    break
            if no_zero_out_n_states:
                ffcycle_end_reason = FFCYCLE_END_REASON.NO_OUTPUT_ZEROES
                break

        ffcycle_stop_timestamp = process_time()
        # Extract ITU's 'forward' value from 'output' neurons
        itu_forward = tuple([n.state for n in self._neurons_o])

        ffcycle_duration_total = ffcycle_stop_timestamp - ffcycle_start_timestamp
        ffcycle_tick_count = tick + 1
        ffcycle_stats = FFCycleStats(
            tick_count=ffcycle_tick_count,
            end_reason=ffcycle_end_reason,
            duration_total=ffcycle_duration_total,
            duration_tick=ffcycle_duration_total / ffcycle_tick_count,
        )

        return itu_forward, ffcycle_stats

    def _get_storage_fname_stem(self, version_history_index: int = -1) -> str:
        """Generate 'stem' of the ITU storage filename.

        :param version_history_index:   index of a target ITU version record in the 'version history'
        :return:                        storage filename's 'stem'
        """
        if version_history_index == -1:
            version_string = self.meta.version_string
        else:
            version_string = str(self.meta.version_history[version_history_index])

        stem = f"{self.__class__.__name__}-{self.meta.uuid.hex}-{version_string}"

        return stem

    def get_storage_fname(self, version_history_index: int = -1) -> str:
        """Generate ITU storage filename.

        :param version_history_index:   index of a target ITU version record in the 'version history'
        :return:                        storage filename
        """
        storage_fname_stem = self._get_storage_fname_stem(version_history_index=version_history_index)
        storage_fname = f"{storage_fname_stem}.sqlite3"

        return storage_fname

    @classmethod
    def load_meta(cls, storage_fpath: Path, app_name: str, app_version: str) -> Any:
        """Load metadata of a preserved ITU instance.

        :param storage_fpath:   file path of a target ITU storage
        :param app_name:        'controller' (creator) application's name
        :param app_version:     'controller' (creator) application's version
        :return:                ITU metadata object
        :raise:                 AigarthITUError, if operation fails
        """
        lm_prefix = f"{cls.__name__}: Load instance metadata: "

        try:
            storage = SQLiteKVDB(
                db_ref=DBReference(scheme=DB_SCHEME.SQLITE3, path=str(storage_fpath)),
                app_codename=app_name,
                app_version=app_version,
            )
        except SQLiteKVDBError as e:
            raise AigarthITUError(f"{lm_prefix}{e.__class__.__name__}: {e}") from e

        if meta := storage.get("meta"):
            return meta
        else:
            raise AigarthITUError(f"{lm_prefix}Metadata not found: {str(storage_fpath)}")

    @classmethod
    def load(cls, storage_fpath: Path, app_name: str, app_version: str) -> Any:
        """Load a preserved ITU instance.

        :param storage_fpath:   file path of a target ITU storage
        :param app_name:        'controller' (creator) application's name
        :param app_version:     'controller' (creator) application's version
        :return:                ITU object
        :raise:                 AigarthITUError, if operation fails
        """
        lm_prefix = f"{cls.__name__}: Load instance: "

        try:
            storage = SQLiteKVDB(
                db_ref=DBReference(scheme=DB_SCHEME.SQLITE3, path=str(storage_fpath)),
                app_codename=app_name,
                app_version=app_version,
            )
        except SQLiteKVDBError as e:
            raise AigarthITUError(f"{lm_prefix}{e.__class__.__name__}: {e}") from e

        if meta := storage.get("meta"):
            if meta.type == cls.__name__:
                return storage["object"]
            else:
                raise AigarthITUError(f"{lm_prefix}Type mismatch: {meta.type} != {cls.__name__}: {str(storage_fpath)}")
        else:
            raise AigarthITUError(f"{lm_prefix}Metadata not found: {str(storage_fpath)}")

    def save(self, storage_dpath: Path, app_name: str, app_version: str) -> Path:
        """Save an ITU instance to a persistent storage.

        :param storage_dpath:   storage directory path
        :param app_name:        'controller' (creator) application's name
        :param app_version:     'controller' (creator) application's version
        :return:                ITU storage file path
        :raise:                 AigarthITUError, if operation fails
        """
        storage_fpath = Path(storage_dpath, self.get_storage_fname())

        try:
            storage = SQLiteKVDB(
                db_ref=DBReference(scheme=DB_SCHEME.SQLITE3, path=str(storage_fpath)),
                auto_commit=False,
                app_codename=app_name,
                app_version=app_version,
            )

            storage["meta"] = self.meta
            storage["object"] = self
            storage.close()
        except SQLiteKVDBError as e:
            raise AigarthITUError(f"{self.lm_prefix}Save instance: {e.__class__.__name__}: {e}") from e

        return storage_fpath

    def reflect(self, *args, **kwargs) -> Any:
        """Encode-transform-decode a single object of outer space."""
        raise NotImplementedError("reflect() not implemented.")

    def reflect_many(self, *args, **kwargs) -> Any:
        """Encode-transform-decode multiple objects of outer space."""
        raise NotImplementedError("reflect_many() not implemented.")
